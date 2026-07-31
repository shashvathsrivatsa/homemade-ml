use crate::*;
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine, Rectangle};

const X_BOUNDS: [f64; 2] = [-6.0, 6.0];
const Y_BOUNDS: [f64; 2] = [-0.5, 3.0];
const CART_Y: f64 = 0.0;
const CART_WIDTH: f64 = 1.2;
const CART_HEIGHT: f64 = 0.45;
const WHEEL_RADIUS: f64 = 0.24;
const POLE_LENGTH: f64 = 1.8;

pub struct CartPoleVisualizaiton {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    cart_x: f32,
    pole_angle: f32,
    episode_len: usize,
    frame_interval: Duration,
    last_render: Option<Instant>,
    cleaned_up: bool,
}

impl CartPoleVisualizaiton {
    pub fn new(fps: u32) -> std::io::Result<Self> {
        if fps == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cart pole visualization FPS must be greater than zero",
            ));
        }

        enable_raw_mode()?;
        let mut output = stdout();
        if let Err(error) = execute!(output, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(error);
        }

        let terminal = match Terminal::new(CrosstermBackend::new(output)) {
            Ok(terminal) => terminal,
            Err(error) => {
                let _ = disable_raw_mode();
                let _ = execute!(stdout(), LeaveAlternateScreen);
                return Err(error);
            }
        };

        Ok(Self {
            terminal,
            cart_x: 0.0,
            pole_angle: 0.0,
            episode_len: 0,
            frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
            last_render: None,
            cleaned_up: false,
        })
    }

    pub fn update(
        &mut self,
        cart_x: f32,
        pole_angle: f32,
        episode_len: usize,
    ) -> std::io::Result<bool> {
        self.cart_x = cart_x;
        self.pole_angle = pole_angle;
        self.episode_len = episode_len;

        let render_due = self
            .last_render
            .is_none_or(|last_render| last_render.elapsed() >= self.frame_interval);
        if render_due {
            self.render()?;
            self.last_render = Some(Instant::now());
        }

        self.check_quit()
    }

    fn render(&mut self) -> std::io::Result<()> {
        let cart_x = self.cart_x as f64;
        let pole_angle = self.pole_angle as f64;
        let pivot_y = CART_Y + CART_HEIGHT;
        let pole_tip_x = cart_x + POLE_LENGTH * pole_angle.sin();
        let pole_tip_y = pivot_y + POLE_LENGTH * pole_angle.cos();

        self.terminal.draw(|frame| {
            let areas = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(5), Constraint::Length(1)])
                .split(frame.area());
            let x_span = X_BOUNDS[1] - X_BOUNDS[0];
            let y_span = Y_BOUNDS[1] - Y_BOUNDS[0];
            let x_scale = 2.0 * f64::from(areas[0].width.max(1)) / x_span;
            let y_scale = 4.0 * f64::from(areas[0].height.max(1)) / y_span;
            let wheel_radius_y = WHEEL_RADIUS * x_scale / y_scale;
            let wheel_y = CART_Y - wheel_radius_y;
            let ground_y = wheel_y - wheel_radius_y;

            let canvas = Canvas::default()
                .marker(symbols::Marker::Braille)
                .x_bounds(X_BOUNDS)
                .y_bounds(Y_BOUNDS)
                .paint(|context| {
                    context.draw(&CanvasLine::new(
                        X_BOUNDS[0],
                        ground_y,
                        X_BOUNDS[1],
                        ground_y,
                        Color::DarkGray,
                    ));
                    context.draw(&Rectangle {
                        x: cart_x - CART_WIDTH / 2.0,
                        y: CART_Y,
                        width: CART_WIDTH,
                        height: CART_HEIGHT,
                        color: Color::Cyan,
                    });
                    for wheel_x in [cart_x - CART_WIDTH / 3.0, cart_x + CART_WIDTH / 3.0] {
                        const SEGMENTS: usize = 32;
                        for segment in 0..SEGMENTS {
                            let angle_1 = std::f64::consts::TAU * segment as f64 / SEGMENTS as f64;
                            let angle_2 =
                                std::f64::consts::TAU * (segment + 1) as f64 / SEGMENTS as f64;
                            context.draw(&CanvasLine::new(
                                wheel_x + WHEEL_RADIUS * angle_1.cos(),
                                wheel_y + wheel_radius_y * angle_1.sin(),
                                wheel_x + WHEEL_RADIUS * angle_2.cos(),
                                wheel_y + wheel_radius_y * angle_2.sin(),
                                Color::White,
                            ));
                        }
                    }
                    context.draw(&CanvasLine::new(
                        cart_x,
                        pivot_y,
                        pole_tip_x,
                        pole_tip_y,
                        Color::Yellow,
                    ));
                });

            let status = Paragraph::new(format!(
                "episode length: {}    x: {:+.3}    angle: {:+.2}°    q: quit",
                self.episode_len,
                self.cart_x,
                self.pole_angle.to_degrees(),
            ))
            .alignment(Alignment::Center);

            frame.render_widget(canvas, areas[0]);
            frame.render_widget(status, areas[1]);
        })?;

        Ok(())
    }

    fn check_quit(&mut self) -> std::io::Result<bool> {
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
                && (key.code == KeyCode::Char('q')
                    || key.code == KeyCode::Char('c')
                        && key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL))
            {
                self.cleaned_up = true;
                let _ = disable_raw_mode();
                let _ = execute!(stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl StateVisualization for CartPoleVisualizaiton {
    fn update_state(
        &mut self,
        cart_x: f32,
        pole_angle: f32,
        episode_len: usize,
        _done: bool,
    ) -> std::io::Result<bool> {
        self.update(cart_x, pole_angle, episode_len)
    }
}

impl Drop for CartPoleVisualizaiton {
    fn drop(&mut self) {
        if self.cleaned_up {
            return;
        }
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}
