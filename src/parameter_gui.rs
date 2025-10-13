use bevy::{prelude::*};
use bevy_egui::{egui, EguiContexts};
use crate::ParticleConfig;

#[repr(C)]
#[derive(Resource, Clone, Copy)]
pub struct GUIConfig
{
    pub fixed_delta_time: f32,          // 4 bytes
    pub gravity: f32,                   // 4 bytes
    pub damping_factor: f32,            // 4 bytes

    pub smoothing_radius: f32,          // 4 bytes
    pub max_energy: f32,                // 4 bytes
    pub target_density: f32,            // 4 bytes
    pub pressure_multiplier: f32,       // 4 bytes
         
    pub viscocity_strength: f32,        // 4 bytes
    pub near_density_multiplier: f32,   // 4 bytes
    
    pub applied_changes: bool,          
}

pub fn gui_system(
    mut contexts: EguiContexts,
    mut gui_config: ResMut<GUIConfig>,
) -> Result
{
    let ctx = contexts.ctx_mut()?;
    egui::Window::new("Sim Params")
        .collapsible(true)
        .resizable(true)
        .default_pos([ctx.screen_rect().width() - 310.0, 10.0])  // Upper right corner
        .show(ctx, |ui: &mut egui::Ui| {
            let mut changed = false;
            changed |= ui.add(egui::Slider::new(&mut gui_config.fixed_delta_time, 0.001..=0.01)
                .text("Fixed Delta Time")
                .step_by(0.001)).changed();
            changed |= ui.add(egui::Slider::new(&mut gui_config.gravity, 0.0..=1000.0)
                .text("Gravity")
                .step_by(1.0)).changed();
            changed |= ui.add(egui::Slider::new(&mut gui_config.damping_factor, 0.0..=1.0)
                .text("Damping Factor")
                .step_by(0.1)).changed();
            changed |= ui.add(egui::Slider::new(&mut gui_config.smoothing_radius, 0.0..=30.0)
                .text("Smoothing Radius")
                .step_by(1.0)).changed();
            changed |= ui.add(egui::Slider::new(&mut gui_config.max_energy, 100.0..=5000.0)
                .text("Max Energy")).changed();
            changed |= ui.add(egui::Slider::new(&mut gui_config.target_density, 0.0..=0.1)
                .text("Target Density")
                .step_by(0.001)).changed();
            changed |= ui.add(egui::Slider::new(&mut gui_config.pressure_multiplier, 1.0..=100000.0)
                .text("Pressure Multiplier")
                .logarithmic(true)
                .smallest_positive(1.0)
                .largest_finite(100_000.0)).changed();
            changed |= ui.add(egui::Slider::new(&mut gui_config.viscocity_strength, 0.0..=10.0)
                .text("Viscocity Strength")).changed();
            changed |= ui.add(egui::Slider::new(&mut gui_config.near_density_multiplier, 1.0..=10000.0)
                .text("Near Density Multiplier")
                .logarithmic(true)
                .smallest_positive(1.0)
                .largest_finite(10_000.0)).changed();
            
            if changed {
                gui_config.applied_changes = true;
            }
        });
    Ok(())
}
pub fn apply_gui_updates(
    mut sim_config: ResMut<ParticleConfig>,
    mut gui_config: ResMut<GUIConfig>,
)
{
    if gui_config.applied_changes && gui_config.is_changed() {
        sim_config.fixed_delta_time = gui_config.fixed_delta_time;
        sim_config.gravity = gui_config.gravity;
        sim_config.damping_factor = gui_config.damping_factor;
        sim_config.smoothing_radius = gui_config.smoothing_radius;
        sim_config.max_energy = gui_config.max_energy;
        sim_config.target_density = gui_config.target_density;
        sim_config.pressure_multiplier = gui_config.pressure_multiplier;
        sim_config.viscocity_strength = gui_config.viscocity_strength;
        sim_config.near_density_multiplier = gui_config.near_density_multiplier;
        
        gui_config.applied_changes = false;
    }
}