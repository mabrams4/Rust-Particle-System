/* ----------------------------------- STRUCTS -----------------------------------*/
struct Config {
    particle_count: u32,            // 4 bytes
    particle_size: f32,             // 4 bytes
    smoothing_radius: f32,          // 4 bytes
    max_energy: f32,                // 4 bytes

    damping_factor: f32,            // 4 bytes
    pad1: f32,                      // 4 bytes
    pad2: f32,                      // 4 bytes
    pad3: f32,                      // 4 bytes

    delta_time: f32,                // 4 bytes
    fixed_delta_time: f32,          // 4 bytes
    frame_count: u32,               // 4 bytes
    gravity: f32,                   // 4 bytes

    target_density: f32,            // 4 bytes
    pressure_multiplier: f32,       // 4 bytes
    viscocity_strength: f32,        // 4 bytes
    near_density_multiplier: f32,   // 4 bytes

    screen_bounds: vec4<f32>,       // 16 bytes     [x_min, x_max, y_min, y_max]
    view_proj: mat4x4<f32>,         // 64 bytes
}

struct SortingParams
{
    n: u32,
    group_width: u32,
    group_height: u32,
    step_index: u32
}

struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color: vec4<f32>,
}

/* ----------------------------------- BINDINGS -----------------------------------*/
@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> config: Config;

@group(0) @binding(2)
var<uniform> sorting_params: SortingParams;

@group(0) @binding(3) 
var<storage, read_write> spatial_lookup: array<vec2<u32>>;

@group(0) @binding(4) 
var<storage, read_write> spatial_lookup_offsets: array<u32>;

@group(0) @binding(5) 
var<storage, read_write> particle_densities: array<vec2<f32>>;  // density, near_density

@group(0) @binding(6) 
var<storage, read_write> predicted_positions: array<vec2<f32>>;

/* --------------------------------- CONSTANTS ---------------------------------*/
const PI: f32 = 3.14159;
const SHADER_DELAY: u32 = 5u;

/* --------------------------------- MISC FUNCTIONS ---------------------------------*/
fn check_screen_bounds(i: u32) 
{
    let x_min = config.screen_bounds[0];
    let x_max = config.screen_bounds[1];
    let y_min = config.screen_bounds[2];
    let y_max = config.screen_bounds[3];

    let pos = particles[i].position;

    if (pos.x >= x_max || pos.x <= x_min) {
        particles[i].velocity.x *= -config.damping_factor;
    }
    if (pos.y >= y_max || pos.y <= y_min) {
        particles[i].velocity.y *= -config.damping_factor;
    }
}

fn set_color(i: u32) 
{
    let speed_sq = dot(particles[i].velocity, particles[i].velocity);
    let energy = 0.5 * 1.0 * speed_sq;

    let normalized = clamp(energy / config.max_energy, 0.0, 1.0);

    var rgb: vec3<f32>;
    if (normalized < 0.5) {
        let t = normalized * 2.0;
        rgb = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), t); // Blue → Green
    } else {
        let t = (normalized - 0.5) * 2.0;
        rgb = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), t); // Green → Red
    }

    particles[i].color = vec4(rgb, 1.0);
}

/* --------------------------------- SPATIAL LOOKUP FUNCTIONS ---------------------------------*/
fn particle_position_to_cell_coord(i: u32) -> vec2<i32>
{
    let x_max = config.screen_bounds[1];
    let y_max = config.screen_bounds[3];

    let cell_x = i32((particles[i].position.x + x_max) / f32(config.smoothing_radius));
    let cell_y = i32((particles[i].position.y + y_max) / f32(config.smoothing_radius));

    return vec2(cell_x, cell_y);
}

fn hash_cell(cell_x: i32, cell_y: i32) -> u32
{
    let a = u32(cell_x) * 15823u; 
    let b = u32(cell_y) * 9737333u; 
    return a + b;
}

fn get_key_from_hash(hash_value: u32) -> u32
{
    return hash_value % config.particle_count;
}

/* --------------------------------- KERNEL FUNCTIONS ---------------------------------*/
fn density_kernel(distance: f32) -> f32
{
    let radius = config.smoothing_radius;
    if (distance >= radius) { return 0f; }
    
    // Normalization factor for 2D Spiky kernel with power 2
    let norm = 10f / (PI * pow(radius, 5f));
    let v = radius - distance;
    return norm * v * v;
}

fn density_kernel_derivative(distance: f32) -> f32
{
    let radius = config.smoothing_radius;
    if (distance >= radius) { return 0f; }
    
    let norm = 10f / (PI * pow(radius, 5f));
    let v = radius - distance;
    return -2f * norm * v;
}

fn near_density_kernel(distance: f32) -> f32
{
    let radius = config.smoothing_radius;
    if (distance >= radius) { return 0f; }
    
    // Normalization factor for 2D Spiky kernel with power 3
    let norm = 15f / (PI * pow(radius, 6f));
    let v = radius - distance;
    return norm * v * v * v;
}

fn near_density_kernel_derivative(distance: f32) -> f32
{
    let radius = config.smoothing_radius;
    if (distance >= radius) { return 0f; }
    
    let norm = 15f / (PI * pow(radius, 6f));
    let v = radius - distance;
    return -3f * norm * v * v;
}

fn viscosity_kernel(distance: f32) -> f32
{
    let radius = config.smoothing_radius;
    if (distance >= radius) { return 0f; }
    
    // Normalization factor for 2D Poly6 kernel
    let norm = 4f / (PI * pow(radius, 8f));
    let v = radius * radius - distance * distance;
    return norm * v * v * v;
}

/* --------------------------------- CALCULATE FUNCTIONS ---------------------------------*/
fn density_to_pressure(density: f32) -> f32
{
    return (density - config.target_density) * config.pressure_multiplier;
}

fn density_to_near_pressure(near_density: f32) -> f32
{
    return near_density * config.near_density_multiplier;
}

var<private> GRID_OFFSETS: array<vec2<i32>, 9> = array<vec2<i32>, 9>(
    vec2<i32>(-1,-1), vec2<i32>(-1,0), vec2<i32>(-1,1),
    vec2<i32>( 0,-1), vec2<i32>( 0,0), vec2<i32>( 0,1),
    vec2<i32>( 1,-1), vec2<i32>( 1,0), vec2<i32>( 1,1)
);

fn calculate_density(curr_particle_index: u32) -> vec2<f32>
{
    var density = 0f;
    var near_density = 0f;

    let x_max = config.screen_bounds[1];
    let y_max = config.screen_bounds[3];

    let curr_particle = particles[curr_particle_index];
    let curr_particle_position = predicted_positions[curr_particle_index];

    let cell_x = i32((curr_particle_position.x + x_max) / f32(config.smoothing_radius));
    let cell_y = i32((curr_particle_position.y + y_max) / f32(config.smoothing_radius));

    let sqr_radius = config.smoothing_radius * config.smoothing_radius;

    for (var i: u32; i < 9u; i++)
    {
        let offset = GRID_OFFSETS[i];
        let neighbor_cell_x = cell_x + offset.x;
        let neighbor_cell_y = cell_y + offset.y;

        let curr_cell_key = get_key_from_hash(hash_cell(neighbor_cell_x, neighbor_cell_y));
        let start_idx = spatial_lookup_offsets[curr_cell_key];   // calculate start idx of this cell key within spatial lookup

        // loop through neighboring particles
        for (var i: u32 = start_idx; i < config.particle_count; i++)
        {
            // break when we reach a new cell key
            let other_particle_cell_key = spatial_lookup[i][0];
            if (other_particle_cell_key != curr_cell_key) { break; }

            let other_particle_index = spatial_lookup[i][1];
            let other_particle_position = predicted_positions[other_particle_index];

            let delta = curr_particle_position - other_particle_position;
            let sqr_distance = dot(delta, delta);

            // skip if particle not within squared radius
            if (sqr_distance > sqr_radius) { continue; }

            let distance = sqrt(sqr_distance);
            density += density_kernel(distance);
            near_density += near_density_kernel(distance);
        }
    }
    return vec2(density, near_density);
}

fn calculate_pressure_force(curr_particle_index: u32) -> vec2<f32>
{
    var pressure_force = vec2(0f, 0f);

    let x_max = config.screen_bounds[1];
    let y_max = config.screen_bounds[3];

    let curr_particle = particles[curr_particle_index];
    let curr_particle_position = predicted_positions[curr_particle_index];

    let cell_x = i32((curr_particle_position.x + x_max) / f32(config.smoothing_radius));
    let cell_y = i32((curr_particle_position.y + y_max) / f32(config.smoothing_radius));

    let sqr_radius = config.smoothing_radius * config.smoothing_radius;

    for (var i: u32; i < 9u; i++)
    {
        let offset = GRID_OFFSETS[i];
        let neighbor_cell_x = cell_x + offset.x;
        let neighbor_cell_y = cell_y + offset.y;

        let curr_cell_key = get_key_from_hash(hash_cell(neighbor_cell_x, neighbor_cell_y));
        let start_idx = spatial_lookup_offsets[curr_cell_key];   // calculate start idx of this cell key within spatial lookup

        // loop through neighboring particles
        for (var i: u32 = start_idx; i < config.particle_count; i++)
        {
            // break when we reach a new cell key
            let other_particle_cell_key = spatial_lookup[i][0];
            if (other_particle_cell_key != curr_cell_key) { break; }

            // skip if comparing particle against itself
            let other_particle_index = spatial_lookup[i][1];
            if (other_particle_index == curr_particle_index) { continue; }

            let delta = predicted_positions[other_particle_index] - curr_particle_position;
            let sqr_distance = dot(delta, delta);

            // skip if particle not within sqr radius
            if (sqr_distance > sqr_radius) { continue; }
            let distance = sqrt(sqr_distance);

            var direction: vec2<f32>;
            if (distance > 0.0001f) {  // Small epsilon to avoid division by zero
                direction = delta / distance;  // Proper normalization
            } else {
                // Particles are essentially at the same position
                // Use a default upward direction to separate them
                direction = vec2(0f, 1f);
            }

            let density = particle_densities[curr_particle_index][0];
            let near_density = particle_densities[curr_particle_index][1];

            let pressure = density_to_pressure(density);
            let near_pressure = density_to_near_pressure(near_density);

            let neighbor_density = particle_densities[other_particle_index][0];
            if (neighbor_density == 0f) { continue; }
            let neighbor_near_density = particle_densities[other_particle_index][1];

            let neighbor_pressure = density_to_pressure(neighbor_density);
            let neighbor_near_pressure = density_to_near_pressure(neighbor_near_density);

            let shared_pressure = (pressure + neighbor_pressure) * 0.5;
            let shared_near_pressure = (near_pressure + neighbor_near_pressure) * 0.5;

            pressure_force += (direction * shared_pressure * density_kernel_derivative(distance)) / neighbor_density;
            pressure_force += (direction * shared_near_pressure * near_density_kernel_derivative(distance)) / neighbor_near_density;
        }
    }
    return pressure_force;
}

fn calculate_viscocity(curr_particle_index: u32) -> vec2<f32>
{
    var viscocity = vec2(0f, 0f);

    let x_max = config.screen_bounds[1];
    let y_max = config.screen_bounds[3];

    let curr_particle = particles[curr_particle_index];
    let curr_particle_position = predicted_positions[curr_particle_index];

    let cell_x = i32((curr_particle_position.x + x_max) / f32(config.smoothing_radius));
    let cell_y = i32((curr_particle_position.y + y_max) / f32(config.smoothing_radius));

    let sqr_radius = config.smoothing_radius * config.smoothing_radius;

    for (var i: u32; i < 9u; i++)
    {
        let offset = GRID_OFFSETS[i];
        let neighbor_cell_x = cell_x + offset.x;
        let neighbor_cell_y = cell_y + offset.y;

        let curr_cell_key = get_key_from_hash(hash_cell(neighbor_cell_x, neighbor_cell_y));
        let start_idx = spatial_lookup_offsets[curr_cell_key];   // calculate start idx of this cell key within spatial lookup

        // loop through neighboring particles
        for (var i: u32 = start_idx; i < config.particle_count; i++)
        {
            // break when we reach a new cell key
            let other_particle_cell_key = spatial_lookup[i][0];
            if (other_particle_cell_key != curr_cell_key) { break; }

            // skip if comparing particle against itself
            let other_particle_index = spatial_lookup[i][1];
            if (other_particle_index == curr_particle_index) { continue; }

            let other_particle = particles[other_particle_index];

            let delta = curr_particle_position - predicted_positions[other_particle_index];
            let sqr_distance = dot(delta, delta);

            // skip if particle not within sqr radius
            if (sqr_distance > sqr_radius) { continue; }

            let distance = sqrt(sqr_distance);
            viscocity += (other_particle.velocity - curr_particle.velocity) * viscosity_kernel(distance);
        }
    }
    return viscocity;
}

fn update_particle_density(i: u32)
{
    let density = calculate_density(i);
    particle_densities[i] = density;
}

fn update_particle_positions(i: u32)
{
    particles[i].position += particles[i].velocity * config.delta_time;
}

fn apply_gravity(i: u32)
{
    particles[i].velocity += vec2(0.0, -config.gravity) * config.delta_time;
}

fn update_predicted_positions(i: u32)
{
    predicted_positions[i] = particles[i].position + particles[i].velocity * config.fixed_delta_time;
}

fn apply_pressure_force(i: u32)
{
    let pressure_force = calculate_pressure_force(i);
    particles[i].velocity += pressure_force / particle_densities[i][0] * config.delta_time;
}

fn apply_viscocity_force(i: u32)
{
    let viscocity_force = calculate_viscocity(i);
    particles[i].velocity += viscocity_force * config.viscocity_strength * config.delta_time;
}

/* ----------------------------------- ENTRY POINT FUNCTIONS -----------------------------------*/
@compute @workgroup_size(64, 1, 1)
fn pre_simulation_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) {
        return;
    }
    //if (config.frame_count < SHADER_DELAY) { return; }

    apply_gravity(i);
    
    update_predicted_positions(i);

    update_particle_density(i);
}

@compute @workgroup_size(64, 1, 1)
fn simulation_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    if (config.frame_count < SHADER_DELAY) { return; }

    apply_pressure_force(i);

    apply_viscocity_force(i);

    update_particle_positions(i);

    check_screen_bounds(i);
    
    set_color(i);
}

@compute @workgroup_size(64, 1, 1)
fn bin_particles_in_grid(@builtin(global_invocation_id) id: vec3<u32>)
{
    let i = id.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    let cell = particle_position_to_cell_coord(i);
    let cell_key = get_key_from_hash(hash_cell(cell.x, cell.y));
    
    spatial_lookup[i] = vec2(cell_key, i);
    spatial_lookup_offsets[i] = 0xFFFFFFFFu; // placeholder 
}

@compute @workgroup_size(64, 1, 1)
fn sort_particles(@builtin(global_invocation_id) id: vec3<u32>) 
{
    let i = id.x;
    if (i >= sorting_params.n / 2u) { return; }

    let group_width = sorting_params.group_width;
    let group_height = sorting_params.group_height;
    let step_index = sorting_params.step_index;

    let h_index = i & (group_width - 1u);
    let index_left = h_index + (group_height + 1u) * (i / group_width);
    var right_step_size: u32;  
    if (step_index == 0u)
    {
        right_step_size = group_height - 2u * h_index;
    }
    else 
    {
        right_step_size = (group_height + 1u) / 2u;
    }
    let index_right = index_left + right_step_size;
    
    // Exit if out of bounds (for non-power of 2 input sizes)
	if (index_right >= sorting_params.n) { return; }

    let value_left = spatial_lookup[index_left][0];
    let value_right = spatial_lookup[index_right][0];

    if (value_left > value_right)
    {
        let temp = spatial_lookup[index_left];
        spatial_lookup[index_left] = spatial_lookup[index_right];
        spatial_lookup[index_right] = temp;
    }
}

@compute @workgroup_size(64, 1, 1)
fn calculate_spatial_lookup_offsets(@builtin(global_invocation_id) id: vec3<u32>)
{
    let i = id.x;
    if (i >= config.particle_count) { return; }

    let key = spatial_lookup[i][0];
    var key_prev = 0xFFFFFFFFu;
    
    if (i > 0u)
    {
        key_prev = spatial_lookup[i - 1u][0];
    }

    if (key != key_prev)
    {
        spatial_lookup_offsets[key] = i;
    }
}



