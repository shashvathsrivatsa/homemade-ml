use crate::*;

// ——— Data loader ————————————————————————————————————————————————————————————————————————————————————————————————————

pub fn load_bw_png(path: String) -> Vec<f32> {
    let img = ImageReader::open(path).unwrap().decode().unwrap();
    let gray = img.to_luma8();
    let pixels: Vec<f32> = gray.pixels().map(|p| p.0[0] as f32 / 255.0).collect();

    pixels
}

pub fn load_data(data_type: DataType) -> (Vec<Vec<f32>>, Vec<f32>) {
    println!("Loading data...");

    let mut xs: Vec<Vec<f32>> = vec![];
    let mut ys: Vec<f32> = vec![];

    for num in 0..=9 {
        let num_of_items = fs::read_dir(format!("data/{}/{}", data_type.get_path(), num))
            .unwrap()
            .count();
        for file in 0..num_of_items {
            xs.push(load_bw_png(format!("data/train/{}/{}.png", num, file)));
            ys.push(num as f32);
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
        }
        .to_string()
    }
}
