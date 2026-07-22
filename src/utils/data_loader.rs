use crate::*;


// ——— Data loader ————————————————————————————————————————————————————————————————————————————————————————————————————

pub fn load_bw_png(path: String) -> Vec<f64> {
    let img = ImageReader::open(path).unwrap().decode().unwrap();
    let gray = img.to_luma8();
    let pixels: Vec<f64> = gray.pixels()
        .map(|p| p.0[0] as f64 / 255.0)
        .collect();

    pixels
}

pub fn load_data(data_type: DataType) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut xs: Vec<Vec<f64>> = vec![];
    let mut ys: Vec<f64> = vec![];

    for num in 0..=9 {
        let num_of_items = fs::read_dir(format!("data/{}/{}", data_type.get_path(), num)).unwrap().count();
        for file in 0..num_of_items {
            xs.push(load_bw_png(format!("data/train/{}/{}.png", num, file)));
            ys.push(num as f64);
        }
    }

    (xs, ys)
}

pub enum DataType {
    Train,
    Test,
}

pub use DataType::*;

impl DataType {
    fn get_path(&self) -> String {
        match self {
            Train => "train",
            Test => "test",
        }.to_string()
    }
}

