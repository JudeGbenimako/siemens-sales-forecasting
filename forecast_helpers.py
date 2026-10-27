import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from bayes_opt import BayesianOptimization
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose




def missing_data(input_data):
    '''
    This function returns dataframe with information about the percentage of nulls in each column and the column data type.

    input: pandas df
    output: pandas df

    '''

    total = input_data.isnull().sum()
    percent = (input_data.isnull().sum() / input_data.isnull().count() * 100)
    table = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in input_data.columns:
        dtype = str(input_data[col].dtype)
        types.append(dtype)
    table["Types"] = types
    return (pd.DataFrame(table))


def mape(actual, pred):
    '''
    Mean Absolute Percentage Error (MAPE) Function

    input: list/series for actual values and predicted values
    output: mape value
    '''
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def plot_sales_distribution(sales_data, product_column="Mapped_GCK", sales_column="Sales_EUR", custom_green="#13b0a6"):
    products = sales_data[product_column].unique()
    n_products = len(products)

    cols = 4
    rows = (n_products + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.5 * rows), facecolor='white')
    axes = axes.flatten()

    for idx, product in enumerate(products):
        ax = axes[idx]
        sns.boxplot(
            x=sales_column,
            data=sales_data[sales_data[product_column] == product],
            ax=ax,
            color=custom_green,
            linewidth=1.5,
            fliersize=5,
            flierprops=dict(markerfacecolor=custom_green, markeredgecolor='black'),
            boxprops=dict(edgecolor='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='black'),
            orient="h"
        )

        ax.set_title(f"Product {product}", fontsize=10, color='black', pad=10)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.3, color='black')

        for spine in ax.spines.values():
            spine.set_edgecolor('black')

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Sales Distribution per Product", fontsize=14, weight='bold', color='black', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #plt.savefig('1', transparent=True, dpi=300, bbox_inches='tight')
    plt.show()


def decompose_and_plot_sales(sales_data, product_col="Mapped_GCK", sales_col="Sales_EUR", date_col="DATE", min_data_points=24, freq="M", line_color_rgb=(0, 153, 153)):
    line_color = [x / 255 for x in line_color_rgb]  # Convert RGB to Matplotlib format

    sales_data_monthly = sales_data.groupby([product_col, pd.Grouper(freq=freq)])[sales_col].sum().reset_index()

    unique_products = sales_data[product_col].unique()
    for product in unique_products:
        product_sales = sales_data_monthly[sales_data_monthly[product_col] == product].set_index(date_col)

        if len(product_sales) >= min_data_points:
            try:
                decomposition = seasonal_decompose(product_sales[sales_col], model="additive", period=12)

                fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
                fig.patch.set_facecolor('white')  # Set figure background to white

                axes[0].plot(product_sales.index, product_sales[sales_col], label="Original Sales", color=line_color)
                axes[0].set_title(f"{product} - Original Sales Data", color="black")
                axes[0].tick_params(colors="black")

                axes[1].plot(decomposition.trend, label="Trend", color=line_color)
                axes[1].set_title("Trend Component", color="black")
                axes[1].tick_params(colors="black")

                axes[2].plot(decomposition.seasonal, label="Seasonality", color=line_color)
                axes[2].set_title("Seasonality Component", color="black")
                axes[2].tick_params(colors="black")

                axes[3].plot(decomposition.resid, label="Residuals", color=line_color)
                axes[3].set_title("Residuals (Unexplained Variations)", color="black")
                axes[3].tick_params(colors="black")

                for ax in axes:
                    ax.legend(facecolor="white", edgecolor="black", fontsize="10")  # Set legend background to white
                    ax.set_facecolor("white")  # Set subplot background to white
                    for spine in ax.spines.values():
                        spine.set_edgecolor("black")

                plt.tight_layout()
                #plt.savefig(f"qualquer coisa {product}", transparent=True, dpi=300, bbox_inches='tight')
                plt.show()
            except ValueError as e:
                print(f"Error decomposing {product}: {e}")
        else:
            print(f"Insufficient data points for decomposition of {product}. Required: {min_data_points}, Found: {len(product_sales)}")



def plot_total_sales_per_product(sales_data, date_col='DATE', product_col='Mapped_GCK', sales_col='Sales_EUR', 
                                 title="Total Sales per Product", color="#13b0a6"):
    # Prepare total sales per product
    agg_sales = sales_data.groupby([date_col, product_col]) \
        .agg({sales_col: 'sum'}) \
        .reset_index() \
        .sort_values([product_col, date_col])

    total_sales_by_product = agg_sales.groupby(product_col)[sales_col].sum().sort_values(ascending=False)

    # Set up plot
    plt.style.use("default")  # Use default (white) theme
    fig, ax = plt.subplots(figsize=(14, 8))  # Larger figure size
    fig.patch.set_facecolor('white')  # Set figure background to white
    ax.set_facecolor('white')  # Set axes background to white

    # Bar plot with clean aesthetics
    bars = sns.barplot(
        x=total_sales_by_product.values,
        y=total_sales_by_product.index,
        color=color,
        ax=ax,
        edgecolor="black",  # Black edges for contrast
        linewidth=1.5       # Slightly thicker bar edges
    )

    # Annotate values with improved text size and style
    for index, value in enumerate(total_sales_by_product.values):
        ax.text(
            value + 0.01 * value,  # Position text slightly to the right of the bar
            index,                 # Align text with the bar
            f"{value:,.0f} €",     # Format value with commas
            va='center',           # Center text vertically
            color='black',         # Text color changed to black
            fontsize=12,           # Larger font size
            fontweight='bold',     # Bold text
            fontfamily='sans-serif'  # Use a clean sans-serif font
        )

    # Clean styling
    ax.set_title(
        title,
        fontsize=20,              
        color='black',  # Changed title color to black
        weight='bold',
        pad=20                     
    )
    ax.set_xlabel(
        "Sales (€)",
        fontsize=14,               
        color='black',  # Changed label color to black
        labelpad=10                
    )
    ax.set_ylabel(
        "Product",
        fontsize=14,               
        color='black',  # Changed label color to black
        labelpad=10                
    )
    ax.tick_params(
        colors='black',  # Changed tick label color to black
        labelsize=12               
    )
    ax.grid(False)  # Remove grid lines

    # Remove spines for a cleaner look
    sns.despine(left=True, bottom=True)

    # Adjust layout for better spacing
    plt.tight_layout()
    #plt.savefig('2', transparent=True, dpi=300, bbox_inches='tight')    # Show the plot
    plt.show()
    



# Function to check stationarity and collect results
def check_stationarity(series, product):
    result = adfuller(series.dropna())
    return {
        "Product": product,
        "ADF Statistic": round(result[0], 4),
        "p-value": round(result[1], 4),
        "Stationary": "Yes" if result[1] < 0.05 else "No"
    }

def plot_acf_pacf_for_group(sales_df, group_name, products, default_lags=24):
    print(f"\n Plotting ACF/PACF for {group_name}...")

    for product in products:
        product_series = sales_df[sales_df["Mapped_GCK"] == product].sort_values("Month")["Sales_EUR"].dropna()
        n_obs = len(product_series)
        max_lag = min(default_lags, n_obs // 2 - 1)

        print(f"Product: {product} - Obs: {n_obs}, Max Lag: {max_lag}")

        if max_lag < 1:
            print(f"⚠️ Not enough data for ACF/PACF for {product}")
            continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(f"{product} - {group_name} - ACF & PACF", fontsize=14)

        plot_acf(product_series, lags=max_lag, ax=axes[0])
        axes[0].set_title("Autocorrelation (ACF)")

        plot_pacf(product_series, lags=max_lag, ax=axes[1], method="ywm")
        axes[1].set_title("Partial Autocorrelation (PACF)")

        plt.tight_layout()
        #plt.savefig('4', transparent=True, dpi=300, bbox_inches='tight')
        plt.show()

def add_lags(df, target, lags_list):
    for lag in lags_list:
        df[f"Sales_Lag_{lag}"] = df.groupby("Mapped_GCK")[target].shift(lag)

        # Fill missing values at the start of each group with 0.
        df[f"Sales_Lag_{lag}"] = df[f"Sales_Lag_{lag}"].fillna(0)

    return df

def add_rolling(df, target, windows):
    # Create rolling features for each window size
    for window in windows:
        # Rolling mean
        df[f"RollingMean_{window}"] = (df
            .groupby("Mapped_GCK")[target]
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True))

        # Fill missing values at the start of each group with 0.
        df[f"RollingMean_{window}"] = df[f"RollingMean_{window}"].fillna(0)

        if window == 3:
            # Rolling standard deviation
            df[f"RollingStd_{window}"] = (df
            .groupby("Mapped_GCK")[target]
            .rolling(window=window, min_periods=1)
            .std()
            .shift(1)
            .reset_index(level=0, drop=True))

            # Fill missing values at the start of each group with 0.
            df[f"RollingStd_{window}"] = df[f"RollingStd_{window}"].fillna(0)
    return df

def xgb_evaluate(data, features, target, tss,
                  learning_rate=0.1, max_depth=5, n_estimators=100,
                  gamma=0, subsample=0.8, colsample_bytree=0.8,
                  early_stopping_rounds=5):

     rmse_scores = []
     for train_idx, val_idx in tss.split(data):
         train = data.iloc[train_idx].copy()
         val = data.iloc[val_idx].copy()

         X_train, y_train = train[features], train[target]
         X_val, y_val = val[features], val[target]

         dtrain = xgb.DMatrix(X_train, label=y_train)
         dval = xgb.DMatrix(X_val, label=y_val)

         params = {
             'learning_rate': learning_rate,
             'max_depth': int(max_depth),
             'gamma': gamma,
             'subsample': subsample,
             'colsample_bytree': colsample_bytree,
             'objective': 'reg:squarederror',
             'eval_metric': 'rmse',
             'seed': 42,
         }

         model = xgb.train(
             params,
             dtrain,
             num_boost_round=n_estimators,
             evals=[(dval, 'eval')],
             early_stopping_rounds=early_stopping_rounds,
             verbose_eval=False
         )

         y_pred = model.predict(dval)

         if "Log" in target:
             y_pred = np.expm1(y_pred)
             y_val = np.expm1(y_val)

         # Calculate RMSE
         rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))        
     return -np.mean(rmse_scores)


def xgb_evaluate_info_with_predictions(data, features, target, tss,
                                       learning_rate=0.1, max_depth=5, n_estimators=100,
                                       gamma=0, subsample=0.8, colsample_bytree=0.8,
                                       early_stopping_rounds=10):

    rmse_scores_train = []
    rmse_scores_val = []
    fold_predictions = []  # List to store predictions and true values for each fold

    for fold, (train_idx, val_idx) in enumerate(tss.split(data), 1):
        train = data.iloc[train_idx].copy()
        val = data.iloc[val_idx].copy()

        X_train, y_train = train[features], train[target]
        X_val, y_val = val[features], val[target]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'learning_rate': learning_rate,
            'max_depth': int(max_depth),
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42,
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )

        # Predict on training and validation sets
        y_pred_train = model.predict(dtrain)
        y_pred_val = model.predict(dval)

        if "Log" in target:
            y_pred_train = np.expm1(y_pred_train)
            y_train = np.expm1(y_train)

            y_pred_val = np.expm1(y_pred_val)
            y_val = np.expm1(y_val)

        # Calculate RMSE for both sets
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

        rmse_scores_train.append(rmse_train)
        rmse_scores_val.append(rmse_val)

        # Store predictions and true values for this fold
        fold_predictions.append({
            'fold': fold,
            'y_train_true': y_train.values.tolist(),
            'y_train_pred': y_pred_train.tolist(),
            'y_val_true': y_val.values.tolist(),
            'y_val_pred': y_pred_val.tolist(),
        })

    # Return a dictionary with scores and predictions for each fold
    return {
        'rmse_train': rmse_scores_train,
        'rmse_val': rmse_scores_val,
        'fold_predictions': fold_predictions,
    }

def xgb_optimization(data, features, target, tss, init_points=5, n_iter=15):
    def optimized_xgb_evaluate(learning_rate, max_depth, n_estimators, gamma, subsample, colsample_bytree):
        return xgb_evaluate(
            data=data,
            features=features,
            target=target,
            tss=tss,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=int(n_estimators),
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        )

    xgb_bo = BayesianOptimization(
        f=optimized_xgb_evaluate,
        pbounds={
            "learning_rate": (0.05, 0.3),
            "max_depth": (3, 10),
            "n_estimators": (100, 500),
            "gamma": (0, 5),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0)
        },
        random_state=42
    )

    xgb_bo.maximize(init_points=init_points, n_iter=n_iter)
    return xgb_bo

def plot_time_series_folds(df, target, tss):
    fig, axs = plt.subplots(nrows=tss.get_n_splits(), ncols=1, figsize=(14, 16), sharex=True)
    fig.suptitle("Time Series Cross-Validation Folds", fontsize=16)

    for i, (train_idx, test_idx) in enumerate(tss.split(df)):
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]

        axs[i].plot(train_data.index, train_data[target], label="Training", color="royalblue")
        axs[i].plot(test_data.index, test_data[target], label="Test", color="#13b0a6")
        axs[i].axvline(test_data.index[0], linestyle="--", color="black", label="Test Start" if i == 0 else "")
        axs[i].set_title(f"Fold {i+1} - Train: {len(train_data)} samples, Test: {len(test_data)} samples")
        axs[i].legend(loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    #plt.savefig('5', transparent=True, dpi=300, bbox_inches='tight')
    plt.show()


def overfitting_plot(results):
    rmse_train = results['rmse_train']
    rmse_val = results['rmse_val']
    fold_predictions = results['fold_predictions']

    # Plot RMSE scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rmse_train) + 1), rmse_train, marker='o', label='Train RMSE')
    plt.plot(range(1, len(rmse_val) + 1), rmse_val, marker='o', label='Validation RMSE')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('RMSE Scores for Each Fold')
    plt.xticks(range(1, len(rmse_train) + 1))
    plt.legend()
    plt.grid(True)
    #plt.savefig('6', transparent=True, dpi=300, bbox_inches='tight')
    plt.show()