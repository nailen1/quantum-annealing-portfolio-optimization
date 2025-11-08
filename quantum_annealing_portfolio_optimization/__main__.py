from .stock_info import StockInfo
from .portfolio_optimizer import PortfolioOptimizer


if __name__ == '__main__':

    NUM_STOCKS_TO_BUY = 2
    BUDGET = 40

    stock_info = StockInfo()

    portfolio_optimizer = PortfolioOptimizer(stock_info, num_stocks_to_buy=NUM_STOCKS_TO_BUY, budget=BUDGET)
    portfolio_optimizer.process_sampleset()
    