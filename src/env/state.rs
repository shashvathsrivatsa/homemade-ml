use crate::*;

// ——— State ——————————————————————————————————————————————————————————————————————————————————————————————————————————

const WINDOW_SIZE: f32 = 100.0;
const MIN_SNAKE_LEN: usize = 50;
const FRUIT_RADIUS: f32 = 2.5;
const FRUIT_SIZE_GAIN: usize = 20;
const HEADING_CHANGE_AMOUNT: f32 = 5.0;
const SPEED: f32 = 0.1;

const WALLS: &[(f32, f32)] = &[
    (-WINDOW_SIZE/2.0, WINDOW_SIZE/2.0),
    (WINDOW_SIZE/2.0, WINDOW_SIZE/2.0),
    (WINDOW_SIZE/2.0, -WINDOW_SIZE/2.0),
    (-WINDOW_SIZE/2.0, -WINDOW_SIZE/2.0),
    (-WINDOW_SIZE/2.0, WINDOW_SIZE/2.0),
];

const REWARD_FRUIT_INCENTIVE: f32 = 2.0;
const PENALTY_HIT_OBSTACLE: f32 = -1.0;
const REWARD_APPROACHING_FRUIT: f32 = 0.1;


pub struct State {
    pub heading: f32,
    pub fruit_coords: (f32, f32),
    segments: VecDeque<(f32, f32)>,
    size_to_gain: usize,

    // internal tracking
    pub fruits_eaten: usize,
    pub episode_len: usize,
}

impl State {
    pub fn new() -> Self {
        let mut segments = VecDeque::new();
        let fruit_coords = Self::spawn_fruit();
        segments.push_front((0., 0.));
        Self {
            heading: 0.0,
            segments,
            fruit_coords,
            fruits_eaten: 0,
            size_to_gain: MIN_SNAKE_LEN,
            episode_len: 0,
        }
    }

    pub fn spawn_fruit() -> (f32, f32) {
        let mut rng = thread_rng();
        let lower_bound = -WINDOW_SIZE/2.0 + FRUIT_RADIUS / 2.0;
        let upper_bound = WINDOW_SIZE/2.0 - FRUIT_RADIUS / 2.0;
        (rng.gen_range(lower_bound..upper_bound), rng.gen_range(lower_bound..upper_bound))
    }

    pub fn step(&mut self, action: usize, visualization: Option<&mut dyn StateVisualization>) -> StepResult {
        let old_state = self.to_vec();
        self.episode_len += 1;
        let dir = match action {
            0=> -HEADING_CHANGE_AMOUNT,
            1 => 0.0,
            2 => HEADING_CHANGE_AMOUNT,
            _ => panic!("wth action?"),
        };

        // Add new head
        self.heading = clamp_deg(self.heading + dir);
        let (dx, dy): (f32, f32) = polar_to_cart(SPEED, self.heading);
        let head_coords = (self.segments[0].0 + dx, self.segments[0].1 + dy);
        self.segments.push_front(head_coords);

        // Anything happened?
        let mut reward = (old_state[1] - self.to_vec()[1]) * REWARD_APPROACHING_FRUIT;
        let mut done = false;
        let cur_segment: ((f32, f32), (f32, f32)) = (self.segments[0], self.segments[1]);

            // collision with fruit
        if cir_intersect(self.segments[0], self.fruit_coords, FRUIT_RADIUS) {
            reward = REWARD_FRUIT_INCENTIVE;
            self.fruit_coords = Self::spawn_fruit();
            self.size_to_gain += FRUIT_SIZE_GAIN;
            self.fruits_eaten += 1;
        }

            // collision with self
        self.segments.iter().skip(2).collect::<Vec<_>>().windows(2).for_each(|segment| {
            let segment = (*segment[0], *segment[1]);
            if seg_intersect(cur_segment, segment) {
                done = true;
                reward = PENALTY_HIT_OBSTACLE;
            }
        });

            // collision with wall
        WALLS.windows(2).for_each(|segment| {
            let segment = (segment[0], segment[1]);
            if seg_intersect(cur_segment, segment) {
                done = true;
                reward = PENALTY_HIT_OBSTACLE;
            }
        });

        // Pop tail
        if self.size_to_gain == 0 {
            self.segments.pop_back();
        } else {
            self.size_to_gain -= 1;
        }

        // Update, reset, output
        let fruits_eaten = self.fruits_eaten;
        let key_pressed: Option<char> = if let Some(visualization) = visualization {
            visualization
                .update_state(
                    self.to_vec(),
                    &self.segments,
                    self.fruit_coords,
                    fruits_eaten,
                    self.episode_len,
                    done,
                )
                .unwrap()
        } else {
            None
        };

        if done { *self = Self::new() }

        let experience = Experience {
            state: old_state,
            action,
            next_state: self.to_vec(),
            reward,
            done,
        };

        StepResult { key_pressed, experience, fruits_eaten }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let rays = &[-90.0, -45.0, 0.0, 45.0, 90.0];

        // TODO: does it even need this? or is proximity to fruit already handled in the continuous reward
        let polar_to_fruit = polar_to_point(self.segments[0], self.heading, self.fruit_coords);

        let rays_to_wall: Vec<f32> = rays.iter().map(|ray| {
            let heading = clamp_deg(self.heading + ray);
            let wall_segments = WALLS.windows(2).map(|w| (w[0], w[1]));
            raycast(self.segments[0], heading, WINDOW_SIZE * 2.0, wall_segments)
        }).collect();

        let rays_to_body: Vec<f32> = rays.iter().map(|ray| {
            let heading = clamp_deg(self.heading + ray);
            let body_segments = self.segments.iter().skip(2).collect::<Vec<_>>();
            let body_segments = body_segments.windows(2).map(|w| (*w[0], *w[1]));
            raycast(self.segments[0], heading, WINDOW_SIZE * 2.0, body_segments)
        }).collect();

        vec![
            self.heading,
            polar_to_fruit.0,
            polar_to_fruit.1,
            rays_to_wall[0],
            rays_to_wall[1],
            rays_to_wall[2],
            rays_to_wall[3],
            rays_to_wall[4],
            rays_to_body[0],
            rays_to_body[1],
            rays_to_body[2],
            rays_to_body[3],
            rays_to_body[4],
        ]
    }
}

pub struct StepResult {
    pub key_pressed: Option<char>,
    pub experience: Experience,
    pub fruits_eaten: usize,
}
