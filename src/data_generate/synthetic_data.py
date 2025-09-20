import numpy as np
import pandas as pd
import random

num_rows = 3000  
np.random.seed(42)
random.seed(42)

def random_day():
    return random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

def random_time_of_day():
    return random.choice(["Morning","Afternoon","Evening","Night"])

def random_device():
    return random.choice(["Mobile","Desktop","Tablet"])

def random_browser():
    return random.choice(["Chrome","Firefox","Safari","Edge","Opera"])

def random_referral():
    return random.choice(["Direct","Search Engine","Email Campaign","Social Media","Ads"])

def random_location():
    india_places = [
        "Delhi, Delhi","Mumbai, Maharashtra","Pune, Maharashtra","Nagpur, Maharashtra",
        "Bangalore, Karnataka","Mysore, Karnataka","Mangalore, Karnataka",
        "Chennai, Tamil Nadu","Coimbatore, Tamil Nadu","Madurai, Tamil Nadu",
        "Hyderabad, Telangana","Warangal, Telangana","Kolkata, West Bengal",
        "Siliguri, West Bengal","Darjeeling, West Bengal","Ahmedabad, Gujarat",
        "Surat, Gujarat","Vadodara, Gujarat","Jaipur, Rajasthan","Udaipur, Rajasthan",
        "Jodhpur, Rajasthan","Lucknow, Uttar Pradesh","Varanasi, Uttar Pradesh",
        "Kanpur, Uttar Pradesh","Patna, Bihar","Gaya, Bihar","Bhubaneswar, Odisha",
        "Cuttack, Odisha","Ranchi, Jharkhand","Jamshedpur, Jharkhand",
        "Chandigarh, Punjab","Amritsar, Punjab","Ludhiana, Punjab",
        "Indore, Madhya Pradesh","Bhopal, Madhya Pradesh","Guwahati, Assam",
        "Shillong, Meghalaya","Kochi, Kerala","Thiruvananthapuram, Kerala"
    ]
    return random.choice(india_places)

def random_category():
    return random.choice(["Electronics","Clothing","Groceries","Beauty",
                          "Home & Kitchen","Sports","Books","Toys","Automotive"])


session_ids = [f"S{i}" for i in range(1, num_rows+1)]
user_ids = [f"U{random.randint(1,500)}" for _ in range(num_rows)] 

return_users = np.random.choice([0,1], size=num_rows, p=[0.6,0.4])

day_of_week = [random_day() for _ in range(num_rows)]
time_of_day = [random_time_of_day() for _ in range(num_rows)]

session_duration = np.random.randint(60, 3600, size=num_rows)  
num_pages_viewed = np.random.randint(1, 20, size=num_rows)
num_items_carted = np.random.randint(0, 10, size=num_rows)

has_viewed_shipping_info = np.random.choice([0,1], size=num_rows, p=[0.7,0.3])
scroll_depth = np.random.randint(10, 100, size=num_rows) 

cart_value = []
for items in num_items_carted:
    if items == 0:
        cart_value.append(0)
    else:
        prices = np.random.randint(200, 5000, size=items)
        total = sum(prices)
        cart_value.append(total)

discount_applied = np.random.choice([0,1], size=num_rows, p=[0.7,0.3])
shipping_fee = [0 if random.random()<0.4 else random.randint(50,300) for _ in range(num_rows)]
free_shipping_eligible = [1 if fee==0 else 0 for fee in shipping_fee]

device_type = [random_device() for _ in range(num_rows)]
browser = [random_browser() for _ in range(num_rows)]
referral_source = [random_referral() for _ in range(num_rows)]
location = [random_location() for _ in range(num_rows)]

if_payment_page_reached = np.random.choice([0,1], size=num_rows, p=[0.5,0.5])  
most_viewed_category = [random_category() for _ in range(num_rows)]

abandoned = []
for dur, items, disc, ship, view_ship in zip(session_duration, num_items_carted, discount_applied, shipping_fee, has_viewed_shipping_info):
    if items == 0:
        abandoned.append(1)  
    elif ship > 200 and disc == 0:
        abandoned.append(1) 
    elif dur < 120 and view_ship == 0:
        abandoned.append(1) 
    else:
        abandoned.append(0)

df = pd.DataFrame({
    "session_id": session_ids,
    "user_id": user_ids,
    "return_user": return_users,
    "day_of_week": day_of_week,
    "time_of_day": time_of_day,
    "session_duration": session_duration,
    "num_pages_viewed": num_pages_viewed,
    "num_items_carted": num_items_carted,
    "has_viewed_shipping_info": has_viewed_shipping_info,
    "scroll_depth": scroll_depth,
    "cart_value": cart_value,
    "discount_applied": discount_applied,
    "shipping_fee": shipping_fee,
    "free_shipping_eligible": free_shipping_eligible,
    "device_type": device_type,
    "browser": browser,
    "referral_source": referral_source,
    "location": location,
    "if_payment_page_reached": if_payment_page_reached,
    "most_viewed_category": most_viewed_category,
    "abandoned": abandoned
})


df.to_csv("../../data/cart_abandonment_dataset.csv", index=False)
print(df.head(10))
