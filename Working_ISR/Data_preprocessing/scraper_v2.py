from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
import time

# Setup driver
def setup_driver():
    options = Options()
    options.headless = True
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# Scrape reviews
def get_reviews_for_book(driver, url):
    driver.get(url)
    time.sleep(5)
    reviews_list = []
    try:
        review_elements = driver.find_elements(By.CSS_SELECTOR, 'article.ReviewCard')
        for review_element in review_elements[:50]:
            review_text = review_element.find_element(By.CSS_SELECTOR, '.ReviewText__content').text
            reviews_list.append(review_text)
    except Exception as e:
        print(f"Error while fetching reviews for URL: {url}\n{e}")
    return reviews_list

# Load the dataset
df = pd.read_csv('/Users/priyankaaskani/Downloads/ISR_project/merged_books.csv')

# Initialize Selenium WebDriver
driver = setup_driver()

# Collect reviews for each book
df['Reviews'] = df.apply(lambda row: get_reviews_for_book(driver, row['URL']), axis=1)

# Close the WebDriver
driver.quit()

# Explode the DataFrame to have separate rows for each review
df_exploded = df.explode('Reviews').reset_index(drop=True)

# Save to a new CSV file
df_exploded.to_csv('merged_reviews.csv', index=False)
