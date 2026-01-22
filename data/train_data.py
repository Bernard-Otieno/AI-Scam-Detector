# data/train_data.py
import numpy as np

def generate_synthetic_data(num_samples=200):
    # Synthetic messages/labels based on PDF scam categories (0=safe, 1=scam)
    scam_examples = [
        "Your M-PESA suspended! Dial *334# now to reverse or lose funds.",  # Transactional + urgency + USSD
        "You won KSh 500,000 prize! Claim at bit.ly/claimnow.",  # Hoax + reward + link
        "Safaricom alert: Account compromised. Send PIN to 254700123456.",  # Impersonation + threat
        "Emergency: Family in accident. Transfer KSh 10,000 urgently.",  # Extortion + emotional
        "Please note that 8936 points in your Safaricom account expire today. We recommend redeeming them as soon as possible to avoid loss. https://safaricom-co.hair/keep-points",  # Urgency + link  
        "Your (2,899pts ) will expire this month. Please redeem them before the reward expires https://ln.run/wWyDw"
    ]
    safe_examples = [
        "Hi Bernard, meeting at 4 PM in Nairobi?",  # Normal
        "Your order confirmed. Delivery tomorrow.",  # Legit
        "Enjoy  50 MB FREE Nyakua Bonus once you spend your daily target! You have used 13 MB so far, Spend 37 MB more today and get 50 MB Free Data Bonus #MwelekeoniInternet.",  # Legit + reward
        "KES 150.00 paid to YOG INTERNATIONAL LIMITED (Acc 272829) on 21/01/26 at 01:39 PM Ref: UAL6D4ISSI. Enquiries, call 0719088000.",  # Legit + transactional
        "You have received KES 3000.0 from JILLIAN CUNGU. M-PESA Ref TDG9ZJT72R. Transaction Ref No CDG78NOEZD.",  # Legit + transactional
        "You have 10 entries  for Safaricom@25 Promo!\nUse M-PESA, Bonga, Buy bundles for a chance to win 1M! Dial *444*25#, *544*25#, *555*25# to Check Entries or OptOut"


    ]
    
    messages = []
    labels = []
    for _ in range(num_samples // 2):
        messages.extend(scam_examples)
        labels.extend([1] * len(scam_examples))
        messages.extend(safe_examples)
        labels.extend([0] * len(safe_examples))
    
    return messages[:num_samples], np.array(labels[:num_samples])