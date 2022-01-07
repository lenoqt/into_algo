# The maximum-subarray problem

Suppose that you been offered the opportunity to invest in the Volatile Chemical Corp. 
Like the chemicals company procedures, the stock price of the Volatile Chemical Corp 
is rather volatile. You are allowed to buy one unit of stock one time and then sell it 
later datte, buying and selling after the close of trading for the day. To compensate 
for this restriction, you are allowed to learn what the price of the stock will be in 
the future. The goal is to maximize your profit. The following image shows the price 
of the stock over a 17-day period. You may buy the stock at any one time, starting after
day 0, when the price is $100 per share. Of couse, you would want to "buy low, sell high"-
buy at the lowest possible price  within a given period. In the figure the lowest price 
occurs after day 7, which occurs after the highest price, after day 1. 

You might think that you can always maximize profit by either buying at the lowest price
or selling at the highest price. For example in the Figure, we would maximize profit by 
buying at the lowest price, after day 7. If this strategy always worked, then it would be 
easy to determine how to maximize profit: find the highest and lowest  prices, and then work 
left from the highest price to find the lowest prior price, work right from the  lowest price 
to find the highest later price, and take the pair with the greater difference.

![alt text](/home/leno/Documents/into_algo/Chapter4/fig4-1.png "Figure 4-1")
![alt text](/home/leno/Documents/into_algo/Chapter4/fig4-2.png "Figure 4-2")

