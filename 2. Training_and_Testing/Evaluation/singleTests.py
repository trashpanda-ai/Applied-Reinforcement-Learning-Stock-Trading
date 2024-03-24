import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
from Envs.env_stocktrading import StockTradingEnv



def prep(train):
    # Sort the DataFrame by the 'date' column
    train = train.sort_values(by='date')

    # Create a mapping of unique dates to their corresponding index
    date_mapping = {date: idx for idx, date in enumerate(train['date'].unique())}

    # Map the 'date' column to the date_mapping to get the new index
    train['index'] = train['date'].map(date_mapping)

    # Set the new index for the DataFrame
    train.set_index('index', inplace=True, drop=True)
    train = train.fillna(0)
    return train
def directory():
    model_path = '/Training and Testing/Evaluation/model/hourly/geglättetNoSentiment/agent_a2c_daily_smoothed.zip'
    path = '/Generated Sentiment Data in FinRL/Data/hourly_smoothed_training_data.csv'
    SENTIMENT = ['random']
    return path, model_path, SENTIMENT

def main():
    # change the parameters according to the model to be tested
    hourly = True
    sentiment = False
    smoothed = True

    if_using_a2c = True
    if_using_ddpg = False
    if_using_ppo = False
    if_using_td3 = False
    if_using_sac = False

    if 'vix' not in INDICATORS:
        INDICATORS.append('vix')

    path, model_path, SENTIMENT = directory()

    trade = pd.read_csv(path)
    trade = prep(trade)


    for seed in range(1):
        model_name = f"seed{seed}"
        print('Ich bin hier')

        trained_a2c = A2C.load(model_path ) if if_using_a2c else None
        trained_ddpg = DDPG.load(model_path + f"/ddpg/{model_name}") if if_using_ddpg else None
        trained_ppo = PPO.load(model_path + f"/ppo/{model_name}") if if_using_ppo else None
        trained_td3 = TD3.load(model_path + f"/td3/{model_name}") if if_using_td3 else None
        trained_sac = SAC.load(model_path + f"/sac/{model_name}") if if_using_sac else None

        stock_dimension = len(trade.tic.unique())
        state_space = 1 + 4 * stock_dimension + len(INDICATORS) * stock_dimension + len(SENTIMENT) * stock_dimension

        buy_cost_list = sell_cost_list = [0.01] * stock_dimension
        num_stock_shares = [0] * stock_dimension

        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "num_stock_shares": num_stock_shares,
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "sentiment_list": SENTIMENT,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
            "seed": seed,
            "hourly": hourly,
        }
        e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold= 1000, risk_indicator_col='vix', **env_kwargs)

        df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
            model=trained_a2c,
            environment=e_trade_gym) if if_using_a2c else (None, None)

        df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
            model=trained_ddpg,
            environment=e_trade_gym) if if_using_ddpg else (None, None)

        df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
            model=trained_ppo,
            environment=e_trade_gym) if if_using_ppo else (None, None)

        df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
            model=trained_td3,
            environment=e_trade_gym) if if_using_td3 else (None, None)

        df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
            model=trained_sac,
            environment=e_trade_gym) if if_using_sac else (None, None)

        df_result_a2c = (
            df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
            if if_using_a2c
            else None
        )
        df_result_ddpg = (
            df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
            if if_using_ddpg
            else None
        )
        df_result_ppo = (
            df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
            if if_using_ppo
            else None
        )
        df_result_td3 = (
            df_account_value_td3.set_index(df_account_value_td3.columns[0])
            if if_using_td3
            else None
        )
        df_result_sac = (
            df_account_value_sac.set_index(df_account_value_sac.columns[0])
            if if_using_sac
            else None
        )

        result = pd.DataFrame(
            {
                "a2c": df_result_a2c["account_value"] if if_using_a2c else None,
                "ddpg": df_result_ddpg["account_value"] if if_using_ddpg else None,
                "ppo": df_result_ppo["account_value"] if if_using_ppo else None,
                "td3": df_result_td3["account_value"] if if_using_td3 else None,
                "sac": df_result_sac["account_value"] if if_using_sac else None,
            }
        )

        if if_using_a2c:
            agent='a2c'
        elif if_using_ddpg:
            agent = 'ddpg'
        elif if_using_ppo:
            agent = 'ppo'
        elif if_using_td3:
            agent = 'td3'
        elif if_using_sac:
            agent = 'sac'
        else:
            agent = ''

        #result.to_csv(f"result{agent}{seed}.csv")
        result.to_csv(f"result{agent}{seed}{smoothed}.csv")





main()