import pandas as pd
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.DataFrame(columns=["Competitor", "Product", "Sentiment"])


urls = { 
    "Competitor_A": "https://amazon.com",
    "Competitor_B": "https://ebay.com",
   
}


for competitor, url in urls.items():
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    products = soup.find_all("h2", class_="product-title") 
    
    for product in products:
        product_name = product.text.strip()
        data = data.append({"Competitor": competitor, "Product": product_name}, ignore_index=True)


data.dropna(inplace=True)
data['Product'] = data['Product'].str.lower().str.strip()


data['Sentiment'] = data['Product'].apply(lambda x: TextBlob(x).sentiment.polarity)


avg_sentiment = data.groupby('Competitor')['Sentiment'].mean()
print(avg_sentiment)

sns.barplot(x=avg_sentiment.index, y=avg_sentiment.values)
plt.title("Average Sentiment by Competitor")
plt.xlabel("Competitor")
plt.ylabel("Average Sentiment")
plt.show()


data.to_csv("competitor_analysis_data.csv", index=False)
avg_sentiment.to_csv("average_sentiment_by_competitor.csv")