# be-streettweets
Backend repository for Street Tweets, an AI-powered road repair prioritization system with social media integration. 
Deployed with docker to GCP.
Access our website here: https://streettweets.vercel.app/

# Run Locally
1. Clone
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Create .env with your own variables
4. Run
```
uvicorn main:app --host=0.0.0.0 --port=8000
```

