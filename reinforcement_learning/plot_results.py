import json
from agent.utils import plot_checkpoint_performance, plot_model_performance

if __name__ == '__main__':
    out_folder = './models_cartpole_changingenv'

    # with open(f"{out_folder}/total_results_checkpoint_performance.json", "r") as f:
    #     total_results = json.load(f)

    # # All plots
    # plot_checkpoint_performance(total_results, dist_type='0', out_folder=out_folder)
    # plot_checkpoint_performance(total_results, dist_type='5', out_folder=out_folder)

    # # Pair plots
    # plot_checkpoint_performance(total_results, dist_type='0', out_folder=out_folder, model_types=['kan_9', 'mlp_40'], name='small')
    # plot_checkpoint_performance(total_results, dist_type='0', out_folder=out_folder, model_types=['kan_16_16', 'mlp_40_40'], name='medium')
    # plot_checkpoint_performance(total_results, dist_type='0', out_folder=out_folder, model_types=['kan_17_17_17', 'mlp_40_40_40'], name='large')

    # plot_checkpoint_performance(total_results, dist_type='5', out_folder=out_folder, model_types=['kan_9', 'mlp_40'], name='small')
    # plot_checkpoint_performance(total_results, dist_type='5', out_folder=out_folder, model_types=['kan_16_16', 'mlp_40_40'], name='medium')
    # plot_checkpoint_performance(total_results, dist_type='5', out_folder=out_folder, model_types=['kan_17_17_17', 'mlp_40_40_40'], name='large')
    
    with open(f"{out_folder}/total_results_model_performance.json", "r") as f:
        total_results = json.load(f)

    plot_model_performance(total_results, out_folder=out_folder, model_types=['kan_9', 'mlp_40'], name='small')
    plot_model_performance(total_results, out_folder=out_folder, model_types=['kan_16_16', 'mlp_40_40'], name='medium')
    plot_model_performance(total_results, out_folder=out_folder, model_types=['kan_17_17_17', 'mlp_40_40_40'], name='large')