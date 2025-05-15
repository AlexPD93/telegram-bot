from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv
import os
import time

load_dotenv()
TOKEN = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=TOKEN)
index_name = "telegram"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
        ) 
    )   

data = [
    {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
    {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
    {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
    {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
    {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
    {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."},
    {"id": "vec7", "text": "Bananas are a great source of potassium and are often eaten as a breakfast fruit."},
    {"id": "vec8", "text": "The banana plant is actually an herb, not a tree."},
    {"id": "vec9", "text": "Banana bread is a popular baked good made from ripe bananas."},
    {"id": "vec10", "text": "Oranges are citrus fruits known for their vitamin C content."},
    {"id": "vec11", "text": "Orange juice is a common breakfast beverage around the world."},
    {"id": "vec12", "text": "Lemons are sour citrus fruits often used to flavor drinks and dishes."},
    {"id": "vec13", "text": "Lemonade is a refreshing drink made from lemons, water, and sugar."},
    {"id": "vec14", "text": "Grapes can be eaten fresh or used to make wine, juice, and raisins."},
    {"id": "vec15", "text": "Strawberries are red, sweet, and often used in desserts like shortcake."},
    {"id": "vec16", "text": "Blueberries are small, blue fruits rich in antioxidants."},
    {"id": "vec17", "text": "Pineapples have a spiky exterior and sweet, juicy flesh inside."},
    {"id": "vec18", "text": "Mangoes are tropical fruits with a sweet and tangy flavor."},
    {"id": "vec19", "text": "Watermelons are large fruits with a green rind and juicy red interior."},
    {"id": "vec20", "text": "Cherries are small, round fruits that can be sweet or tart."},
    {"id": "vec21", "text": "Python is a popular programming language known for its readability."},
    {"id": "vec22", "text": "The Eiffel Tower is one of the most famous landmarks in Paris."},
    {"id": "vec23", "text": "The theory of relativity was developed by Albert Einstein."},
    {"id": "vec24", "text": "Mount Everest is the highest mountain in the world."},
    {"id": "vec25", "text": "The Great Wall of China is visible from space."},
    {"id": "vec26", "text": "Photosynthesis is the process by which plants make their food."},
    {"id": "vec27", "text": "The capital of Japan is Tokyo."},
    {"id": "vec28", "text": "Shakespeare wrote many famous plays, including Hamlet and Macbeth."},
    {"id": "vec29", "text": "The speed of light is approximately 299,792 kilometers per second."},
    {"id": "vec30", "text": "The human heart pumps blood throughout the body."},
    {"id": "vec31", "text": "World War II ended in 1945."},
    {"id": "vec32", "text": "The Amazon rainforest is the largest tropical rainforest in the world."},
    {"id": "vec33", "text": "The Mona Lisa was painted by Leonardo da Vinci."},
    {"id": "vec34", "text": "Water boils at 100 degrees Celsius at sea level."},
    {"id": "vec35", "text": "The internet has revolutionized global communication."},
    {"id": "vec36", "text": "The Pythagorean theorem relates the sides of a right triangle."},
    {"id": "vec37", "text": "The Pacific Ocean is the largest ocean on Earth."},
    {"id": "vec38", "text": "The periodic table organizes chemical elements by atomic number."},
    {"id": "vec39", "text": "The Statue of Liberty was a gift from France to the United States."},
    {"id": "vec40", "text": "Gravity is the force that attracts objects toward the center of the Earth."}
]

embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

vectors = []
for d, e in zip(data, embeddings):
    vectors.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

index.upsert(
    vectors=vectors,
    namespace="ns1"
)

query = "How do I create a table?"

def query_pinecone(query: str):
    # Query the index
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
        "input_type": "query"
        }
)

    results = index.query(
        namespace="ns1",
        vector=embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    return results