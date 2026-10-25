//! wgpu compute backend: a GPU port of the CPU reference renderer in `renderer.rs`.
//!
//! All scene data is static, so every scene's BVH nodes, 80-byte hot intersection records,
//! cold per-simplex attributes and materials are uploaded once as storage buffers. The only
//! per-frame input is a small uniform block (camera event + tetrad + render settings + the
//! frame's `FrameDynamics`: causal frontier and light-pulse flashes).
//!
//! The kernel (`shaders/render.wgsl`) writes packed `0x00RRGGBB` pixels into a storage buffer
//! which is read back into the caller's framebuffer, so the window/HUD code is identical for
//! both backends and the CPU path stays available as a reference for image-diff tests.

use crate::mesh4d::SurfacePattern;
use crate::minkowski::Vec4;
use crate::renderer::{
    gamma_lut, Camera4D, FrameDynamics, RenderConfig, RenderMode, ShadowMode, SliceMode,
    MAX_FLASHES, MAX_HAZE_ORIGINS,
};
use crate::scene::Scene4D;
use bytemuck::{Pod, Zeroable};
#[cfg(not(target_arch = "wasm32"))]
use wgpu::util::DeviceExt;

pub const WORKGROUP_SIZE: u32 = 8;
pub const RENDER_WGSL: &str = include_str!("shaders/render.wgsl");

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
pub(crate) struct GpuNode {
    pub(crate) min: [f32; 4],
    pub(crate) max: [f32; 4],
    pub(crate) right_or_first: u32,
    pub(crate) count: u32,
    pub(crate) _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuGeom {
    pub(crate) v0: [f32; 4],
    pub(crate) n: [f32; 4],
    pub(crate) b1: [f32; 4],
    pub(crate) b2: [f32; 4],
    pub(crate) b3: [f32; 4],
}

/// Cold attributes, transposed so each row holds one component for all 4 vertices.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuAttr {
    pub(crate) uv_u: [f32; 4],
    pub(crate) uv_v: [f32; 4],
    pub(crate) tb_x: [f32; 4],
    pub(crate) tb_y: [f32; 4],
    pub(crate) tb_z: [f32; 4],
    pub(crate) tau: [f32; 4],
    pub(crate) grad_u: [f32; 4],
    pub(crate) grad_v: [f32; 4],
    pub(crate) normal: [f32; 4],
    pub(crate) four_vel: [f32; 4],
    pub(crate) misc: [f32; 4],
    pub(crate) decided_at: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuMaterial {
    pub(crate) base: [f32; 4],
    pub(crate) checker: [f32; 4],
    pub(crate) params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuUniforms {
    pub(crate) cam_pos: [f32; 4],
    pub(crate) e: [[f32; 4]; 4],
    pub(crate) light_pos: [f32; 4],
    pub(crate) half_wh: [f32; 2],
    pub(crate) size: [u32; 2],
    pub(crate) mode: u32,
    pub(crate) shadow_mode: u32,
    pub(crate) slice_mode: u32,
    pub(crate) num_nodes: u32,
    pub(crate) frontier_origin: [f32; 4],
    pub(crate) frontier_view_offset: f32,
    pub(crate) num_flashes: u32,
    pub(crate) frontier_tau: f32,
    pub(crate) prophecy_delta: f32,
    pub(crate) num_haze: u32,
    pub(crate) _pad: [u32; 3],
    pub(crate) haze_origin: [[f32; 4]; MAX_HAZE_ORIGINS],
    pub(crate) flash_at: [[f32; 4]; MAX_FLASHES],
    pub(crate) flash_col_width: [[f32; 4]; MAX_FLASHES],
}

impl GpuUniforms {
    pub(crate) fn build(
        camera: &Camera4D,
        cfg: &RenderConfig,
        dynamics: &FrameDynamics,
        num_nodes: u32,
        light_pos: [f32; 4],
    ) -> Self {
        let (w, h) = (cfg.width, cfg.height);
        let aspect = w as f32 / h as f32;
        let half_h = (0.5 * camera.fov_y_rad).tan();
        let half_w = aspect * half_h;

        let flashes = dynamics.active_flashes();
        let mut flash_at = [[0.0; 4]; MAX_FLASHES];
        let mut flash_col_width = [[0.0; 4]; MAX_FLASHES];
        for (i, f) in flashes.iter().enumerate() {
            let c = f.color * f.intensity;
            flash_at[i] = v4(f.at);
            flash_col_width[i] = [c.x, c.y, c.z, f.width];
        }
        let frontier = dynamics.frontier_or_camera(camera);
        let haze = dynamics.haze_origins(camera, cfg.slice_mode);
        let mut haze_origin = [[0.0; 4]; MAX_HAZE_ORIGINS];
        for (slot, &o) in haze_origin.iter_mut().zip(&haze) {
            *slot = v4(o);
        }

        Self {
            cam_pos: v4(camera.position),
            e: camera.tetrad.e.map(v4),
            light_pos,
            half_wh: [half_w, half_h],
            size: [w as u32, h as u32],
            mode: match cfg.mode {
                RenderMode::RelativisticShaded => 0,
                RenderMode::LookbackIsochrones => 1,
                RenderMode::Simplex4DInspector => 2,
                RenderMode::DecisionAge => 3,
            },
            shadow_mode: match cfg.shadow_mode {
                ShadowMode::Off => 0,
                ShadowMode::LorentzInvariant => 1,
                ShadowMode::CameraSlice => 2,
            },
            slice_mode: match cfg.slice_mode {
                SliceMode::PastLightCone => 0,
                SliceMode::CameraSimultaneity => 1,
                SliceMode::Frontier => 2,
                SliceMode::ProphecySlice => 3,
            },
            num_nodes,
            frontier_origin: v4(frontier.origin),
            frontier_view_offset: dynamics.frontier_view_offset(),
            num_flashes: flashes.len() as u32,
            frontier_tau: frontier.proper_time,
            prophecy_delta: cfg.prophecy_delta,
            num_haze: haze.len().min(MAX_HAZE_ORIGINS) as u32,
            _pad: [0; 3],
            haze_origin,
            flash_at,
            flash_col_width,
        }
    }
}

/// Storage buffers can't be zero-sized; pad empty arrays with one zeroed element.
fn non_empty<T: Pod + Zeroable>(mut v: Vec<T>) -> Vec<T> {
    if v.is_empty() {
        v.push(T::zeroed());
    }
    v
}

/// Build the shared lookup table buffer (4096-entry gamma LUT + 14 5x7 glyphs).
pub(crate) fn build_tables_data() -> Vec<u32> {
    let mut tables: Vec<u32> = gamma_lut().iter().map(|&b| b as u32).collect();
    for &ch in &SHADER_GLYPHS {
        tables.extend(crate::hud::glyph_5x7(ch).iter().map(|&row| row as u32));
    }
    tables
}

/// Packed GPU storage buffer arrays for a single [`Scene4D`], shared by native `wgpu` and browser WebGPU.
pub(crate) struct PackedSceneGpuData {
    pub(crate) nodes: Vec<GpuNode>,
    pub(crate) num_nodes: u32,
    pub(crate) geoms: Vec<GpuGeom>,
    pub(crate) attrs: Vec<GpuAttr>,
    pub(crate) materials: Vec<GpuMaterial>,
    pub(crate) light_pos: [f32; 4],
}

impl PackedSceneGpuData {
    pub(crate) fn from_scene(scene: &Scene4D) -> Self {
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
                    decided_at: s.decided_at,
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

        let lp = scene.light_pos;
        Self {
            nodes: non_empty(nodes),
            num_nodes,
            geoms: non_empty(geoms),
            attrs: non_empty(attrs),
            materials: non_empty(materials),
            light_pos: [0.0, lp.x, lp.y, lp.z],
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct SceneBuffers {
    bind_group: wgpu::BindGroup,
    num_nodes: u32,
    light_pos: [f32; 4],
}

#[cfg(not(target_arch = "wasm32"))]
struct FrameBuffers {
    width: usize,
    height: usize,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[cfg(not(target_arch = "wasm32"))]
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    scene_layout: wgpu::BindGroupLayout,
    frame_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    tables_buf: wgpu::Buffer,
    scenes: Vec<SceneBuffers>,
    frame: Option<FrameBuffers>,
    adapter_name: String,
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
impl GpuRenderer {
    /// Initialise a GPU device and upload all scenes. Returns an error if no suitable adapter
    /// is available, so callers can fall back to the CPU renderer.
    pub fn new(scenes: &[Scene4D]) -> Result<Self, String> {
        pollster::block_on(Self::new_async(scenes))
    }

    async fn new_async(scenes: &[Scene4D]) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
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
        let tables = build_tables_data();
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
            scene_layout,
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
        let packed = PackedSceneGpuData::from_scene(scene);

        let mk = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let nodes_buf = mk("bvh nodes", bytemuck::cast_slice(&packed.nodes));
        let geoms_buf = mk("hot geoms", bytemuck::cast_slice(&packed.geoms));
        let attrs_buf = mk("cold attrs", bytemuck::cast_slice(&packed.attrs));
        let mats_buf = mk("materials", bytemuck::cast_slice(&packed.materials));

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

        SceneBuffers {
            bind_group,
            num_nodes: packed.num_nodes,
            light_pos: packed.light_pos,
        }
    }

    /// Replace `scenes[scene_idx]`'s GPU buffers (a Dynamic Mode scene grew new geometry).
    pub fn update_scene(&mut self, scene_idx: usize, scene: &Scene4D) {
        self.scenes[scene_idx] = Self::upload_scene(&self.device, &self.scene_layout, scene);
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
        dynamics: &FrameDynamics,
        framebuffer: &mut [u32],
    ) {
        let (w, h) = (cfg.width, cfg.height);
        assert_eq!(framebuffer.len(), w * h, "framebuffer size mismatch");
        self.ensure_frame_buffers(w, h);
        let scene = &self.scenes[scene_idx];
        let frame = self.frame.as_ref().expect("frame buffers initialised");

        let uniforms =
            GpuUniforms::build(camera, cfg, dynamics, scene.num_nodes, scene.light_pos);
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::renderer::PROPHECY_DELTA_DEFAULT;
    use crate::renderer::render_frame;
    use crate::scene::DemoScene;
    use crate::minkowski::Vec3;

    /// The Rust `#[repr(C)]` mirrors must match the WGSL struct layouts byte for byte, or the
    /// shader silently reads garbage. Naga computes the WGSL side from the shader source.
    #[test]
    fn wgsl_struct_layouts_match_rust() {
        use std::mem::{offset_of, size_of};
        use wgpu::naga;

        let module = naga::front::wgsl::parse_str(RENDER_WGSL).expect("render.wgsl parses");
        let wgsl_struct = |name: &str| -> (u32, Vec<(String, u32)>) {
            let ty = module
                .types
                .iter()
                .map(|(_, ty)| ty)
                .find(|ty| ty.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("WGSL struct {name} not found"));
            match &ty.inner {
                naga::TypeInner::Struct { members, span } => (
                    *span,
                    members
                        .iter()
                        .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
                        .collect(),
                ),
                _ => panic!("{name} is not a struct"),
            }
        };
        let check = |name: &str, rust_size: usize, rust_offsets: &[(&str, usize)]| {
            let (span, members) = wgsl_struct(name);
            assert_eq!(span as usize, rust_size, "{name} size");
            for &(field, offset) in rust_offsets {
                let wgsl_offset = members
                    .iter()
                    .find(|(n, _)| n == field)
                    .unwrap_or_else(|| panic!("{name}.{field} missing in WGSL"))
                    .1;
                assert_eq!(wgsl_offset as usize, offset, "{name}.{field} offset");
            }
        };

        check(
            "Uniforms",
            size_of::<GpuUniforms>(),
            &[
                ("light_pos", offset_of!(GpuUniforms, light_pos)),
                ("num_nodes", offset_of!(GpuUniforms, num_nodes)),
                ("frontier_origin", offset_of!(GpuUniforms, frontier_origin)),
                ("frontier_view_offset", offset_of!(GpuUniforms, frontier_view_offset)),
                ("num_flashes", offset_of!(GpuUniforms, num_flashes)),
                ("frontier_tau", offset_of!(GpuUniforms, frontier_tau)),
                ("num_haze", offset_of!(GpuUniforms, num_haze)),
                ("haze_origin", offset_of!(GpuUniforms, haze_origin)),
                ("flash_at", offset_of!(GpuUniforms, flash_at)),
                ("flash_col_width", offset_of!(GpuUniforms, flash_col_width)),
            ],
        );
        check(
            "Attr",
            size_of::<GpuAttr>(),
            &[
                ("misc", offset_of!(GpuAttr, misc)),
                ("decided_at", offset_of!(GpuAttr, decided_at)),
            ],
        );
        check("Node", size_of::<GpuNode>(), &[]);
        check("Geom", size_of::<GpuGeom>(), &[]);
        check("Material", size_of::<GpuMaterial>(), &[]);
    }

    /// Flash lighting through the per-frame uniform block must match the CPU reference. The
    /// flashes are emitted at the camera's position `2R` before the camera event, so on the
    /// Real Camera view their shells light everything at distance `R` in the camera frame.
    #[test]
    fn gpu_matches_cpu_with_flashes() {
        let scenes: Vec<Scene4D> = DemoScene::ALL.iter().map(|&d| Scene4D::build(d)).collect();
        let mut gpu = match GpuRenderer::new(&scenes) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };

        let (w, h) = (192, 120);
        let mut plain_fb = vec![0u32; w * h];
        let mut cpu_fb = vec![0u32; w * h];
        let mut gpu_fb = vec![0u32; w * h];

        for (i, scene) in scenes.iter().enumerate() {
            let cam = Camera4D::new(scene.demo.initial_camera_pos(), 65.0);
            let flash_at = |r: f32| {
                Vec4::from_time_space(cam.position.t - 2.0 * r, cam.position.spatial())
            };
            let dynamics = FrameDynamics {
                frontier: None,
                frontier_lag: 0.0,
                remote_origins: Vec::new(),
                flashes: vec![
                    crate::renderer::FlashEvent {
                        at: flash_at(3.0),
                        width: 1.2,
                        intensity: 2.0,
                        color: Vec3::new(1.0, 0.3, 0.2),
                    },
                    crate::renderer::FlashEvent {
                        at: flash_at(6.0),
                        width: 1.5,
                        intensity: 3.0,
                        color: Vec3::new(0.2, 0.6, 1.0),
                    },
                ],
            };
            for shadow_mode in [ShadowMode::LorentzInvariant, ShadowMode::Off] {
                let cfg = RenderConfig {
                    width: w,
                    height: h,
                    mode: RenderMode::RelativisticShaded,
                    shadow_mode,
                    slice_mode: SliceMode::PastLightCone,
                    prophecy_delta: PROPHECY_DELTA_DEFAULT,
                };
                render_frame(scene, &cam, &cfg, &FrameDynamics::default(), &mut plain_fb);
                render_frame(scene, &cam, &cfg, &dynamics, &mut cpu_fb);
                gpu.render(i, &cam, &cfg, &dynamics, &mut gpu_fb);

                let (_, lit_frac) = diff_stats(&plain_fb, &cpu_fb, 4);
                let (max_diff, bad_frac) = diff_stats(&cpu_fb, &gpu_fb, 4);
                println!(
                    "scene {} flashes/{:?}: {:.1}% px lit, max CPU/GPU diff {}, {:.3}% px > 4",
                    i + 1, shadow_mode, lit_frac * 100.0, max_diff, bad_frac * 100.0
                );
                assert!(lit_frac > 0.01, "scene {}: flashes lit too few pixels", i + 1);
                assert!(
                    bad_frac < 0.005,
                    "scene {} flashes/{:?}: {:.3}% of pixels differ",
                    i + 1, shadow_mode, bad_frac * 100.0
                );
            }
        }
    }

    /// A live Dynamic Mode snapshot (scene 6 after the camera has thrust and coasted for a few
    /// seconds, so tubes end in caps at the frontier) renders the same on both backends in
    /// every view, including the ones that look at the frontier itself.
    #[test]
    fn gpu_matches_cpu_dynamic_world() {
        use crate::dynamic::DynWorld;
        use crate::renderer::Frontier;

        let mut cam = Camera4D::new(DemoScene::RelativisticBilliards.initial_camera_pos(), 65.0);
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        for _ in 0..40 {
            cam.tetrad.boost_local(Vec3::new(0.01, 0.0, 0.01));
            cam.position += cam.tetrad.e[0] * 0.1;
            cam.proper_time += 0.1;
            world.advance(Frontier { origin: cam.position, proper_time: cam.proper_time }, cam.tetrad.e[0]);
        }
        let scene = world.build_scene();
        let dynamics = FrameDynamics {
            frontier: Some(world.frontier),
            flashes: Vec::new(),
            frontier_lag: world.frontier_lag(cam.tetrad.e[0]),
            remote_origins: Vec::new(),
        };
        let mut gpu = match GpuRenderer::new(std::slice::from_ref(&scene)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };

        let (w, h) = (192, 120);
        let mut cpu_fb = vec![0u32; w * h];
        let mut gpu_fb = vec![0u32; w * h];
        for (mode, shadow_mode, slice_mode) in [
            (RenderMode::RelativisticShaded, ShadowMode::LorentzInvariant, SliceMode::PastLightCone),
            (RenderMode::DecisionAge, ShadowMode::Off, SliceMode::PastLightCone),
            (RenderMode::DecisionAge, ShadowMode::Off, SliceMode::Frontier),
            (RenderMode::RelativisticShaded, ShadowMode::LorentzInvariant, SliceMode::Frontier),
            (RenderMode::RelativisticShaded, ShadowMode::Off, SliceMode::ProphecySlice),
        ] {
            let cfg = RenderConfig {
                width: w,
                height: h,
                mode,
                shadow_mode,
                slice_mode,
                prophecy_delta: PROPHECY_DELTA_DEFAULT,
            };
            render_frame(&scene, &cam, &cfg, &dynamics, &mut cpu_fb);
            gpu.render(0, &cam, &cfg, &dynamics, &mut gpu_fb);
            let (max_diff, bad_frac) = diff_stats(&cpu_fb, &gpu_fb, 4);
            println!(
                "dynamic {mode:?}/{shadow_mode:?}/{slice_mode:?}: max diff {max_diff}, {:.3}% px > 4",
                bad_frac * 100.0
            );
            assert!(bad_frac < 0.005, "{slice_mode:?}: {:.3}% of pixels differ", bad_frac * 100.0);
        }
    }

    /// Decision-age shading of geometry stamped with `decided_at` (as Dynamic Mode will emit it)
    /// under an active frontier must match the CPU reference, exercising the `decided_at`
    /// attribute and the `frontier_origin` / `frontier_tau` uniforms.
    #[test]
    fn gpu_matches_cpu_decision_age_with_frontier() {
        use crate::bvh4d::Bvh4D;
        use crate::mesh4d::{
            build_worldtube_simplices, build_worldtube_simplices_with_decided,
            sample_inertial_worldline, Mesh3D,
        };
        use crate::renderer::Frontier;

        // A ground plane (static) and a row of cubes whose worldtube steps were decided at
        // camera proper times 0, 1, 2, ... (spanning several hue bands and contour rings).
        let ground = Mesh3D::ground_grid(22.0, 22.0, -2.2, 4, 4);
        let mut simplices = build_worldtube_simplices(
            &ground,
            &sample_inertial_worldline(Vec3::default(), Vec3::default(), -40.0, 40.0, 1),
            0,
        );
        let cube = Mesh3D::cube(Vec3::new(0.6, 0.6, 0.6));
        for k in 0..4 {
            let pos = Vec3::new(-3.0 + 2.0 * k as f32, -1.0, 4.0);
            let samples =
                sample_inertial_worldline(pos, Vec3::new(0.0, 0.0, -0.3), -20.0, 0.0, 20);
            let decided: Vec<f32> = (0..samples.len()).map(|i| i as f32 + k as f32).collect();
            simplices.extend(build_worldtube_simplices_with_decided(
                &cube,
                &samples,
                Some(&decided),
                1,
            ));
        }
        let scene = Scene4D {
            demo: DemoScene::TwinParadox,
            bvh: Bvh4D::build(simplices),
            materials: Scene4D::common_materials(),
            light_pos: Vec3::new(0.0, 6.0, 2.0),
            t_min: -20.0,
            t_max: 0.0,
        };
        let mut gpu = match GpuRenderer::new(std::slice::from_ref(&scene)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };

        let (w, h) = (192, 120);
        let mut cpu_fb = vec![0u32; w * h];
        let mut gpu_fb = vec![0u32; w * h];
        let cam = Camera4D::new(Vec4::new(0.0, 0.0, 0.0, -4.0), 65.0);
        let dynamics = FrameDynamics {
            frontier: Some(Frontier {
                origin: cam.position + Vec4::new(-0.5, 0.2, 0.0, 0.0),
                proper_time: 24.0,
            }),
            frontier_lag: 0.0,
            // Another player 3 s behind: its cone reaches the God's-eye slice.
            remote_origins: vec![cam.position + Vec4::new(-3.0, 2.0, 0.0, 6.0)],
            flashes: Vec::new(),
        };
        for slice_mode in
            [SliceMode::PastLightCone, SliceMode::CameraSimultaneity, SliceMode::ProphecySlice]
        {
            let cfg = RenderConfig {
                width: w,
                height: h,
                mode: RenderMode::DecisionAge,
                shadow_mode: ShadowMode::Off,
                slice_mode,
                prophecy_delta: PROPHECY_DELTA_DEFAULT,
            };
            render_frame(&scene, &cam, &cfg, &dynamics, &mut cpu_fb);
            gpu.render(0, &cam, &cfg, &dynamics, &mut gpu_fb);
            let distinct: std::collections::HashSet<u32> = cpu_fb.iter().copied().collect();
            let (max_diff, bad_frac) = diff_stats(&cpu_fb, &gpu_fb, 4);
            println!(
                "decision age {:?}: {} distinct colours, max diff {}, {:.3}% px > 4",
                slice_mode, distinct.len(), max_diff, bad_frac * 100.0
            );
            assert!(distinct.len() > 50, "{slice_mode:?}: image is suspiciously flat");
            assert!(bad_frac < 0.005, "{slice_mode:?}: {:.3}% of pixels differ", bad_frac * 100.0);
        }
    }

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
            (RenderMode::DecisionAge, ShadowMode::Off, SliceMode::PastLightCone),
            (RenderMode::DecisionAge, ShadowMode::Off, SliceMode::ProphecySlice),
            (RenderMode::RelativisticShaded, ShadowMode::LorentzInvariant, SliceMode::Frontier),
            (RenderMode::RelativisticShaded, ShadowMode::CameraSlice, SliceMode::ProphecySlice),
            (RenderMode::LookbackIsochrones, ShadowMode::Off, SliceMode::Frontier),
        ];
        let mut cpu_fb = vec![0u32; w * h];
        let mut gpu_fb = vec![0u32; w * h];
        let no_dynamics = FrameDynamics::default();

        for (i, scene) in scenes.iter().enumerate() {
            let mut cam = Camera4D::new(scene.demo.initial_camera_pos(), 65.0);
            // Exercise a boosted, rotated observer too.
            cam.tetrad.boost_local(Vec3::new(0.2, 0.0, 0.6));
            cam.tetrad.rotate_local(0.15, -0.05, 0.0);
            for &(mode, shadow_mode, slice_mode) in &configs {
                let cfg = RenderConfig {
                    width: w,
                    height: h,
                    mode,
                    shadow_mode,
                    slice_mode,
                    prophecy_delta: PROPHECY_DELTA_DEFAULT,
                };
                render_frame(scene, &cam, &cfg, &no_dynamics, &mut cpu_fb);
                gpu.render(i, &cam, &cfg, &no_dynamics, &mut gpu_fb);
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
