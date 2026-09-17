import yaml
from src.data_preparation import load_and_optimize_data, split_features_and_target
from src.model_training import train_and_eval_all_models

def main():
    # 1. Load configuration
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Loading and cleaning data...")
    # 2. Load, clean, and downcast memory
    df = load_and_optimize_data(config['data']['raw_path'])

    # 3. Extract target and split features
    cls_target = config['target']['classification_col']
    X_train, X_test, y_train, y_test = split_features_and_target(
        df, 
        target_col=cls_target, 
        test_size=config['model']['test_size'], 
        random_state=config['model']['random_state']
    )

    print("Training and evaluating 3 models...")
    # 4. Train 3 models and collect performance metrics
    results_df, trained_models = train_and_eval_all_models(
        X_train, X_test, y_train, y_test, config
    )

    # 5. Display comparison table
    print("\n=================== MODEL COMPARISON ===================")
    print(results_df.to_string(index=False))
    print("========================================================\n")

if __name__ == "__main__":
    main()