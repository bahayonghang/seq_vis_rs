use ndarray::{Array3, s};  // 添加 s 宏的导入
use plotters::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use rand::Rng;  // 添加 Rng trait 的导入

// 添加字体设置函数
// 修改 setup_font 函数
fn setup_font() -> FontFamily<'static> {
    let font_families = vec![
        "LXGW WenKai",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "PingFang SC",
    ];

    for font in font_families {
        if let Ok(font_family) = FontFamily::try_from(font) {
            return font_family;
        }
    }

    // 使用正确的枚举变量
    FontFamily::SansSerif
}

#[derive(Debug)]
pub struct PlotConfig {
    pub plot_type: String,
    pub path: PathBuf,
}

pub fn get_plot_config() -> Result<PlotConfig, Box<dyn Error>> {
    println!("Please enter the type of visualization ('train' or 'test'):");
    let mut plot_type = String::new();
    io::stdin().read_line(&mut plot_type)?;
    let plot_type = plot_type.trim().to_string();

    if !["train", "test"].contains(&plot_type.as_str()) {
        return Err("Invalid type. Please use 'train' or 'test'.".into());
    }

    println!("Please enter the absolute path to the result directory:");
    let mut path = String::new();
    io::stdin().read_line(&mut path)?;
    let path = PathBuf::from(path.trim());

    if !path.is_absolute() {
        return Err("The path must be an absolute path.".into());
    }
    if !path.exists() {
        return Err(format!("The path {} does not exist.", path.display()).into());
    }

    Ok(PlotConfig { plot_type, path })
}

pub fn prompt_action() -> Result<String, Box<dyn Error>> {
    println!("Choose an action: continue, reset, exit");
    let mut action = String::new();
    io::stdin().read_line(&mut action)?;
    Ok(action.trim().to_string())
}

pub fn visualize_once(config: &PlotConfig) -> Result<(), Box<dyn Error>> {
    match config.plot_type.as_str() {
        "train" => plot_npy_all_train(&config.path)?,
        "test" => plot_npy_all(&config.path)?,
        _ => return Err("Invalid type".into()),
    }
    Ok(())
}

fn load_name_list(path: &Path) -> Result<Option<Vec<String>>, Box<dyn Error>> {
    let name_list_path = path.join("name_list.txt");
    if name_list_path.exists() {
        let file = File::open(name_list_path)?;
        let reader = BufReader::new(file);
        let names: Result<Vec<_>, _> = reader.lines().collect();
        Ok(Some(names?))
    } else {
        Ok(None)
    }
}

fn create_chart<'a, DB: DrawingBackend + 'a>(
    area: &'a DrawingArea<DB, plotters::coord::Shift>,
    title: &str,
    true_slice: ndarray::ArrayView1<f64>,
    pred_slice: ndarray::ArrayView1<f64>,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    let font = setup_font();

    let min_val = true_slice.iter()
        .chain(pred_slice.iter())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_val = true_slice.iter()
        .chain(pred_slice.iter())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let mut chart = ChartBuilder::on(area)
        .caption(title, (font.clone(), 20))  // 使用中文字体
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0..true_slice.len(),
            *min_val..(*max_val * 1.1)
        )?;

    // 配置网格和标签，使用中文字体
    chart
        .configure_mesh()
        .disable_mesh()
        .x_label_style((font.clone(), 15))  // 设置X轴标签字体
        .y_label_style((font.clone(), 15))  // 设置Y轴标签字体
        .draw()?;

    // Plot true values with thicker lines
    chart
        .draw_series(LineSeries::new(
            true_slice.iter().enumerate().map(|(x, y)| (x, *y)),
            BLUE.mix(0.9),  // 移除 & 和 stroke_width
        ))?
        .label("True")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Plot predicted values with thicker lines
    chart
        .draw_series(LineSeries::new(
            pred_slice.iter().enumerate().map(|(x, y)| (x, *y)),
            RED.mix(0.9),  // 移除 & 和 stroke_width
        ))?
        .label("Pred")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // 配置图例，使用中文字体
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font((font, 15))  // 设置图例字体
        .draw()?;

    Ok(())
}

pub fn plot_npy_all(result_path: &Path) -> Result<(), Box<dyn Error>> {
    let true_path = result_path.join("true.npy");
    let pred_path = result_path.join("pred.npy");

    let true_array: Array3<f64> = ndarray_npy::read_npy(true_path)?;
    let pred_array: Array3<f64> = ndarray_npy::read_npy(pred_path)?;

    println!("true_array shape: {:?}", true_array.shape());
    println!("pred_array shape: {:?}", pred_array.shape());

    let name_list = load_name_list(result_path)?;

    let mut rng = rand::thread_rng();
    let random_index = rng.gen_range(0..true_array.shape()[0]);
    println!("Selected random index: {}", random_index);

    // 增加输出图像的尺寸和DPI
    let output_path = result_path.join("true_pred_all.png");
    let root = BitMapBackend::new(&output_path, (3840, 2160))  // 增加分辨率
        .into_drawing_area();
    root.fill(&WHITE)?;

    let dimensions = true_array.shape()[2];
    let areas = root.split_evenly((
        (dimensions as f64 / 3.0).ceil() as usize,
        std::cmp::min(3, dimensions),
    ));

    for (i, area) in areas.into_iter().take(dimensions).enumerate() {
        let true_slice = true_array.slice(s![random_index, .., i]);
        let pred_slice = pred_array.slice(s![random_index, .., i]);

        let title = if let Some(ref names) = name_list {
            names[i].clone()
        } else {
            format!("维度 {}", i + 1)  // 使用中文
        };

        create_chart(&area, &title, true_slice, pred_slice)?;
    }

    root.present()?;
    println!("Plot saved to {}", output_path.display());

    Ok(())
}

// 同样更新 plot_npy_all_train 函数...
pub fn plot_npy_all_train(result_path: &Path) -> Result<(), Box<dyn Error>> {
    let true_path = result_path.join("true_train.npy");
    let pred_path = result_path.join("pred_train.npy");

    let true_array: Array3<f64> = ndarray_npy::read_npy(true_path)?;
    let pred_array: Array3<f64> = ndarray_npy::read_npy(pred_path)?;

    println!("true_array shape: {:?}", true_array.shape());
    println!("pred_array shape: {:?}", pred_array.shape());

    let name_list = load_name_list(result_path)?;

    let mut rng = rand::thread_rng();
    let random_index = rng.gen_range(0..true_array.shape()[0]);
    println!("Selected random index: {}", random_index);

    // 增加输出图像的尺寸和DPI
    let output_path = result_path.join("true_pred_all_train.png");
    let root = BitMapBackend::new(&output_path, (1920, 1080))  // 增加分辨率
        .into_drawing_area();
    root.fill(&WHITE)?;

    let dimensions = true_array.shape()[2];
    let areas = root.split_evenly((
        (dimensions as f64 / 3.0).ceil() as usize,
        std::cmp::min(3, dimensions),
    ));

    for (i, area) in areas.into_iter().take(dimensions).enumerate() {
        let true_slice = true_array.slice(s![random_index, .., i]);
        let pred_slice = pred_array.slice(s![random_index, .., i]);

        let title = if let Some(ref names) = name_list {
            names[i].clone()
        } else {
            format!("维度 {}", i + 1)  // 使用中文
        };

        create_chart(&area, &title, true_slice, pred_slice)?;
    }

    root.present()?;
    println!("Plot saved to {}", output_path.display());

    Ok(())
}