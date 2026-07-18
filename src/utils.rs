use crate::*;


// ——— Utils ——————————————————————————————————————————————————————————————————————————————————————————————————————————

pub fn range(start: f64, end: f64, step: f64) -> Vec<f64> {
    let mut result = Vec::new();
    let mut current = start;


    while current < end {
        result.push(current);
        current += step;
    }

    result
}


pub fn plot(xs: &Vec<f64>, ys: &Vec<f64>) {
    let root = BitMapBackend::new("plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        // .caption("f(x) = 3x² - 4x + 5", ("sans-serif", 20))      // title
        .margin(20)                                              // border
        .x_label_area_size(30)                                   // space for x labels
        .y_label_area_size(40)                                   // space for y labels
        .build_cartesian_2d(-5f64..5f64, 0f64..110f64)
        .unwrap();

    chart.configure_mesh()
        // .x_desc("x")             // x axis label
        // .y_desc("y")             // y axis label
        .draw()
        .unwrap();

    chart.draw_series(
        LineSeries::new(
            xs.iter().zip(ys.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        )
    ).unwrap();

    root.present().unwrap();
} 

static COUNTER: AtomicUsize = AtomicUsize::new(0);
pub fn next_id() -> usize {
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
