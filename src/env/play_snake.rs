use crate::{SnakeTestGraph, State};
use std::{io, thread, time::{Duration, Instant}};

const TICKS_PER_SECOND: u32 = 240;
const RENDER_FRAMES_PER_SECOND: u32 = 60;

/// Runs Snake with keyboard-controlled steering.
///
/// Hold `a` to turn left or `d` to turn right. Releasing the key goes straight.
/// Press `f` to freeze or resume the frame.
/// Press `q` or Ctrl+C to exit.
pub fn play_snake() -> io::Result<()> {
    let mut state = State::new();
    let mut visualization = SnakeTestGraph::new(RENDER_FRAMES_PER_SECOND)?;
    let tick_duration = Duration::from_secs_f64(1.0 / f64::from(TICKS_PER_SECOND));
    let mut next_tick = Instant::now();
    let mut action = 1;
    let mut frozen = false;
    let mut freeze_key_down = false;

    loop {
        next_tick += tick_duration;
        let key_pressed = if frozen {
            visualization.poll_key()?
        } else {
            let step_result = state.step(action, Some(&mut visualization));
            if step_result.experience.done {
                action = 1;
            }
            step_result.key_pressed
        };

        if matches!(key_pressed, Some('q' | '\u{3}')) {
            return Ok(());
        }

        let freeze_key_is_down = matches!(key_pressed, Some('f' | 'F'));
        if freeze_key_is_down && !freeze_key_down {
            frozen = !frozen;
            action = 1;
        }
        freeze_key_down = freeze_key_is_down;

        action = match key_pressed {
            Some('a' | 'A') => 2,
            Some('d' | 'D') => 0,
            Some(_) => action,
            None => 1,
        };

        let now = Instant::now();
        if next_tick > now {
            thread::sleep(next_tick - now);
        } else {
            next_tick = now;
        }
    }
}
