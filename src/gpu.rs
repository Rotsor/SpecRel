//! wgpu compute backend: a GPU port of the CPU reference renderer in `renderer.rs`.
//!
//! All scene data is static, so every scene's BVH nodes, 80-byte hot intersection records,
//! cold per-simplex attributes and materials are uploaded once as storage buffers. The only
//! per-frame input is a small uniform block (camera event + tetrad + render settings).
//!
//! The kernel (`shaders/render.wgsl`) writes packed `0x00RRGGBB` pixels into a storage buffer
//! which is read back into the caller's framebuffer, so the window/HUD code is identical for
//! both backends and the CPU path stays available as a reference for image-diff tests.

use crate::mesh4d::SurfacePattern;
use crate::minkowski::Vec4;
use crate::renderer::{gamma_lut, Camera4D, RenderConfig, RenderMode, ShadowMode, SliceMode};
use crate::scene::Scene4D;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 8;

/// Glyphs needed by the clock face's digital proper-time readout, in the order the shader
/// expects (`0-9`, then `+`, `-`, `.`, `S`).
const SHADER_GLYPHS: [char; 14] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '.', 'S',
];

#[inline]
fn v4(v: Vec4) -> [f32; 4] {
    [v.t, v.x, v.y, v.z]
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuNode {
    min: [f32; 4],
    max: [f32; 4],
    right_or_first: u32,
    count: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuGeom {
    v0: [f32; 4],
    n: [f32; 4],
    b1: [f32; 4],
    b2: [f32; 4],
    b3: [f32; 4],
}

/// Cold attributes, transposed so each row holds one component for all 4 vertices.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuAttr {
    uv_u: [f32; 4],
    uv_v: [f32; 4],
    tb_x: [f32; 4],
    tb_y: [f32; 4],
    tb_z: [f32; 4],
    tau: [f32; 4],
    grad_u: [f32; 4],
    grad_v: [f32; 4],
    normal: [f32; 4],
    four_vel: [f32; 4],
    misc: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMaterial {
    base: [f32; 4],
    checker: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuUniforms {
    cam_pos: [f32; 4],
    e: [[f32; 4]; 4],
    light_pos: [f32; 4],
    half_wh: [f32; 2],
    size: [u32; 2],
    mode: u32,
    shadow_mode: u32,
    slice_mode: u32,
    num_nodes: u32,
}

struct SceneBuffers {
    bind_group: wgpu::BindGroup,
    num_nodes: u32,
    light_pos: [f32; 4],
}

struct FrameBuffers {
    width: usize,
    height: usize,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    frame_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    tables_buf: wgpu::Buffer,
    scenes: Vec<SceneBuffers>,
    frame: Option<FrameBuffers>,
    adapter_name: String,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Storage buffers can't be zero-sized; pad empty arrays with one zeroed element.
fn non_empty<T: Pod + Zeroable>(mut v: Vec<T>) -> Vec<T> {
    if v.is_empty() {
        v.push(T::zeroed());
    }
    v
}

impl GpuRenderer {
    /// Initialise a GPU device and upload all scenes. Returns an error if no suitable adapter
    /// is available, so callers can fall back to the CPU renderer.
    pub fn new(scenes: &[Scene4D]) -> Result<Self, String> {
        pollster::block_on(Self::new_async(scenes))
    }

    async fn new_async(scenes: &[Scene4D]) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?})", info.name, info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("spec-rel device"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("failed to create GPU device: {e}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
        });

        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
            ],
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[Some(&scene_layout), Some(&frame_layout)],
            ..Default::default()
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<GpuUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Lookup tables: the CPU's gamma LUT verbatim (bit-exact tone mapping), then glyph rows.
        let mut tables: Vec<u32> = gamma_lut().iter().map(|&b| b as u32).collect();
        for &ch in &SHADER_GLYPHS {
            tables.extend(crate::hud::glyph_5x7(ch).iter().map(|&row| row as u32));
        }
        let tables_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tables"),
            contents: bytemuck::cast_slice(&tables),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let scenes = scenes
            .iter()
            .map(|s| Self::upload_scene(&device, &scene_layout, s))
            .collect();

        Ok(Self {
            device,
            queue,
            pipeline,
            frame_layout,
            uniform_buf,
            tables_buf,
            scenes,
            frame: None,
            adapter_name,
        })
    }

    fn upload_scene(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        scene: &Scene4D,
    ) -> SceneBuffers {
        let bvh = &scene.bvh;

        let nodes: Vec<GpuNode> = bvh
            .nodes()
            .iter()
            .map(|n| GpuNode {
                min: v4(n.aabb.min),
                max: v4(n.aabb.max),
                right_or_first: n.right_or_first,
                count: n.count,
                _pad: [0; 2],
            })
            .collect();
        let num_nodes = nodes.len() as u32;

        let geoms: Vec<GpuGeom> = bvh
            .hot_geoms()
            .iter()
            .map(|g| GpuGeom {
                v0: v4(g.v0),
                n: v4(g.hyperplane_normal),
                b1: v4(g.dual_b1),
                b2: v4(g.dual_b2),
                b3: v4(g.dual_b3),
            })
            .collect();

        let attrs: Vec<GpuAttr> = bvh
            .simplices
            .iter()
            .map(|s| {
                let col = |f: &dyn Fn(usize) -> f32| [f(0), f(1), f(2), f(3)];
                GpuAttr {
                    uv_u: col(&|i| s.uv[i].0),
                    uv_v: col(&|i| s.uv[i].1),
                    tb_x: col(&|i| s.tri_bary[i].x),
                    tb_y: col(&|i| s.tri_bary[i].y),
                    tb_z: col(&|i| s.tri_bary[i].z),
                    tau: s.proper_time,
                    grad_u: col(&|i| s.tau_uv_grad[i].0),
                    grad_v: col(&|i| s.tau_uv_grad[i].1),
                    normal: v4(s.surface_normal_4d),
                    four_vel: v4(s.four_velocity),
                    misc: [
                        s.uv_world_scale.0,
                        s.uv_world_scale.1,
                        f32::from_bits(s.material_id as u32),
                        0.0,
                    ],
                }
            })
            .collect();

        let materials: Vec<GpuMaterial> = scene
            .materials
            .iter()
            .map(|m| {
                let pattern: u32 = match m.pattern {
                    SurfacePattern::Checkerboard => 0,
                    SurfacePattern::ClockFace => 1,
                    SurfacePattern::LightClock => 2,
                };
                GpuMaterial {
                    base: [m.base_color.x, m.base_color.y, m.base_color.z, m.emission],
                    checker: [
                        m.checker_color.x,
                        m.checker_color.y,
                        m.checker_color.z,
                        m.grid_scale,
                    ],
                    params: [m.wireframe_width, f32::from_bits(pattern), m.cavity_length, 0.0],
                }
            })
            .collect();

        let mk = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let nodes_buf = mk("bvh nodes", bytemuck::cast_slice(&non_empty(nodes)));
        let geoms_buf = mk("hot geoms", bytemuck::cast_slice(&non_empty(geoms)));
        let attrs_buf = mk("cold attrs", bytemuck::cast_slice(&non_empty(attrs)));
        let mats_buf = mk("materials", bytemuck::cast_slice(&non_empty(materials)));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: nodes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: geoms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: attrs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mats_buf.as_entire_binding(),
                },
            ],
        });

        let lp = scene.light_pos;
        SceneBuffers {
            bind_group,
            num_nodes,
            light_pos: [0.0, lp.x, lp.y, lp.z],
        }
    }

    /// Human-readable adapter description, e.g. `"NVIDIA GeForce RTX 4070 (Vulkan)"`.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    fn ensure_frame_buffers(&mut self, width: usize, height: usize) {
        if let Some(f) = &self.frame {
            if f.width == width && f.height == height {
                return;
            }
        }
        let size = (width * height * 4) as u64;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output pixels"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame bind group"),
            layout: &self.frame_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tables_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        self.frame = Some(FrameBuffers {
            width,
            height,
            output,
            readback,
            bind_group,
        });
    }

    /// Render `scenes[scene_idx]` (as passed to [`GpuRenderer::new`]) into `framebuffer`
    /// (`cfg.width * cfg.height` packed `0x00RRGGBB` pixels), matching `renderer::render_frame`.
    pub fn render(
        &mut self,
        scene_idx: usize,
        camera: &Camera4D,
        cfg: &RenderConfig,
        framebuffer: &mut [u32],
    ) {
        let (w, h) = (cfg.width, cfg.height);
        assert_eq!(framebuffer.len(), w * h, "framebuffer size mismatch");
        self.ensure_frame_buffers(w, h);
        let scene = &self.scenes[scene_idx];
        let frame = self.frame.as_ref().expect("frame buffers initialised");

        let aspect = w as f32 / h as f32;
        let half_h = (0.5 * camera.fov_y_rad).tan();
        let half_w = aspect * half_h;
        let uniforms = GpuUniforms {
            cam_pos: v4(camera.position),
            e: camera.tetrad.e.map(v4),
            light_pos: scene.light_pos,
            half_wh: [half_w, half_h],
            size: [w as u32, h as u32],
            mode: match cfg.mode {
                RenderMode::RelativisticShaded => 0,
                RenderMode::LookbackIsochrones => 1,
                RenderMode::Simplex4DInspector => 2,
            },
            shadow_mode: match cfg.shadow_mode {
                ShadowMode::Off => 0,
                ShadowMode::LorentzInvariant => 1,
                ShadowMode::CameraSlice => 2,
            },
            slice_mode: match cfg.slice_mode {
                SliceMode::PastLightCone => 0,
                SliceMode::CameraSimultaneity => 1,
            },
            num_nodes: scene.num_nodes,
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("render pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &scene.bind_group, &[]);
            pass.set_bind_group(1, &frame.bind_group, &[]);
            pass.dispatch_workgroups(
                (w as u32).div_ceil(WORKGROUP_SIZE),
                (h as u32).div_ceil(WORKGROUP_SIZE),
                1,
            );
        }
        let size = (w * h * 4) as u64;
        encoder.copy_buffer_to_buffer(&frame.output, 0, &frame.readback, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = frame.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| {
            r.expect("failed to map GPU readback buffer")
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("GPU poll failed");
        {
            let view = slice
                .get_mapped_range()
                .expect("readback buffer not mapped");
            framebuffer.copy_from_slice(bytemuck::cast_slice(&view));
        }
        frame.readback.unmap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::render_frame;
    use crate::scene::DemoScene;
    use crate::minkowski::Vec3;

    /// Per-channel absolute difference statistics between two packed RGB images.
    fn diff_stats(a: &[u32], b: &[u32], tol: u32) -> (u32, f64) {
        let mut max_diff = 0;
        let mut bad = 0usize;
        for (&pa, &pb) in a.iter().zip(b) {
            let d = (0..3)
                .map(|c| {
                    let ca = (pa >> (8 * c)) & 0xFF;
                    let cb = (pb >> (8 * c)) & 0xFF;
                    ca.abs_diff(cb)
                })
                .max()
                .unwrap();
            max_diff = max_diff.max(d);
            if d > tol {
                bad += 1;
            }
        }
        (max_diff, bad as f64 / a.len() as f64)
    }

    /// The GPU backend must reproduce the CPU reference renderer. Floating-point evaluation
    /// order differs (FMA, `dot` ordering, transcendental precision), which can flip a handful
    /// of pixels exactly on texture/edge thresholds, so we allow a tiny fraction of outliers.
    #[test]
    fn gpu_matches_cpu_reference() {
        let scenes: Vec<Scene4D> = DemoScene::ALL.iter().map(|&d| Scene4D::build(d)).collect();
        let mut gpu = match GpuRenderer::new(&scenes) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };

        let (w, h) = (192, 120);
        let configs = [
            (RenderMode::RelativisticShaded, ShadowMode::LorentzInvariant, SliceMode::PastLightCone),
            (RenderMode::RelativisticShaded, ShadowMode::CameraSlice, SliceMode::CameraSimultaneity),
            (RenderMode::LookbackIsochrones, ShadowMode::Off, SliceMode::PastLightCone),
            (RenderMode::Simplex4DInspector, ShadowMode::Off, SliceMode::PastLightCone),
        ];
        let mut cpu_fb = vec![0u32; w * h];
        let mut gpu_fb = vec![0u32; w * h];

        for (i, scene) in scenes.iter().enumerate() {
            let mut cam = Camera4D::new(scene.demo.initial_camera_pos(), 65.0);
            // Exercise a boosted, rotated observer too.
            cam.tetrad.boost_local(Vec3::new(0.2, 0.0, 0.6));
            cam.tetrad.rotate_local(0.15, -0.05, 0.0);
            for &(mode, shadow_mode, slice_mode) in &configs {
                let cfg = RenderConfig { width: w, height: h, mode, shadow_mode, slice_mode };
                render_frame(scene, &cam, &cfg, &mut cpu_fb);
                gpu.render(i, &cam, &cfg, &mut gpu_fb);
                let (max_diff, bad_frac) = diff_stats(&cpu_fb, &gpu_fb, 4);
                println!(
                    "scene {} {:?}/{:?}/{:?}: max channel diff {}, {:.3}% px > 4",
                    i + 1, mode, shadow_mode, slice_mode, max_diff, bad_frac * 100.0
                );
                assert!(
                    bad_frac < 0.005,
                    "scene {} {:?}/{:?}/{:?}: {:.3}% of pixels differ",
                    i + 1, mode, shadow_mode, slice_mode, bad_frac * 100.0
                );
            }
        }
    }
}
