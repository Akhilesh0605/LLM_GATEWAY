from app.models import ComplexityTier

def classify(query : str) -> tuple[int,ComplexityTier]:
    #scoring based on length
    token_score=0
    length=len(query.split())
    if(length) < 10 : token_score+=1
    elif(10<length<30) : token_score+=3
    else : token_score+=5

    #keyword detection & technical terms & structure 
    keyword_score=0
    terms =0
    technical_score=0
    structure_score=0
    complex_keywords=[
        "design", "implement", "compare", "difference between",
        "write code", "architecture", "optimize", "explain why",
        "how does", "build", "system", "distributed"
    ]

    simple_keywords=[
        "what is", "define", "who is", "when did",
        "what are", "list", "name"
    ]

    technical_terms = [
    "api", "database", "algorithm", "cache", "distributed",
    "async", "latency", "throughput", "scalability", "microservice"
    ]

    query_lower=query.lower()
    for keyword in complex_keywords:
        if keyword in query_lower:   
            keyword_score += 3
    for keyword in simple_keywords:
        if keyword in query_lower:
            keyword_score-=2
    for term in technical_terms:
        if term in query_lower: terms+=1

    if terms==0 : technical_score=0
    elif terms<3 : technical_score=2
    else : technical_score=4

    questions = query.count("?")

    if questions == 0 : structure_score=1
    elif questions == 1 : structure_score=-1
    else : structure_score=2

    score=token_score+keyword_score+technical_score+structure_score

    final_score=max(1,min(10,score)) 


    if 1<=final_score<=3 : tier = ComplexityTier.SIMPLE
    elif 4<=final_score<=6 : tier= ComplexityTier.MEDIUM
    else : tier=ComplexityTier.COMPLEX

    return (final_score,tier)