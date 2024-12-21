use clap::{Parser, Subcommand};
use seq_vis_rs::{get_plot_config, prompt_action, visualize_once};
use std::process;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Visualize {
        #[arg(short, long)]
        interactive: bool,
    },
}

fn run_interactive_mode() {
    loop {
        match get_plot_config() {
            Ok(config) => {
                match visualize_once(&config) {
                    Ok(()) => {
                        match prompt_action() {
                            Ok(action) => match action.as_str() {
                                "continue" | "reset" => continue, // 继续下一次循环
                                "exit" => break, // 退出循环
                                _ => {
                                    println!("Invalid action. Exiting.");
                                    break;
                                }
                            },
                            Err(e) => {
                                eprintln!("Error reading action: {}", e);
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error during visualization: {}", e);
                        break;
                    }
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }

}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Visualize { interactive } => {
            if interactive {
                run_interactive_mode();
            } else {
                println!("Please provide plot type (train/test) and path");
                process::exit(1);
            }
        }
    }
}