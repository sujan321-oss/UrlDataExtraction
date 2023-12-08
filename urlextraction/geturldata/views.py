from django.shortcuts import render,redirect
from django.http import JsonResponse
import json
from django.http import HttpResponse

from django.shortcuts import render
from django.http import JsonResponse
from urlextraction.settings import BASE_DIR
import os

import langchain 
from langchain.prompts import PromptTemplate
import uuid

from langchain.document_loaders import WebBaseLoader


def get_data(request):
    content=""
    if request.method == "POST":
        try:
            data = request.POST['data']
            data=json.loads(data)
            
            if data['url1']!="" and data['url2']!="":
                urls=[
                    data['url1'],
                    data['url2']
                ]
                
            elif data["url1"]=="" and data["ulr2"]!="":
                  urls=[
                      data['url2']
                  ]
                
            elif data["url2"]=="" and data["url1"]!="":
                urls=[
                    data['url1']
                ]
            
            
            try:
                loader=WebBaseLoader(urls)
                web_content=loader.load()
                unique_identifier=str(uuid.uuid4())
                response = JsonResponse({"message":"data stored sucessfully"})
                response.set_cookie("unique_identifier", unique_identifier)
                 
                dir=os.path.join(BASE_DIR,f"urlcontent/{unique_identifier}.txt")
                
                for i in web_content:
                    content+=i.page_content
                          
                # print(dir)
                with open(dir,'w') as file:
                    file.write(content)
                
                return response
              
            except Exception as e:
                print(e)
                return JsonResponse({"message":"error_occured"})
           
            return JsonResponse({"message": "url are sucessfully sored"})
        
        
        except Exception as e:
            print(e)
            return JsonResponse({"message":"Error occured"})

    return render(request, "extracor.html")






from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import numpy as np
import faiss
import os
from langchain.llms import GooglePalm
from langchain.chains import LLMChain


def question_answer(request):
    os.environ['GOOGLE_API_KEY']="AIzaSyA3OFkfvPoy5aLkiJxGTrw_xHqMjJYheL0"
    content=""
    if request.method=="POST":
        data = request.POST['data']
        data=json.loads(data)
        question=data['question']
 
        
        # print("quesrion is >>>>",question)
        # question=data['question']
        # print(question)
    
        
        
        unique_identifier=request.COOKIES.get("unique_identifier",None)
        print(unique_identifier)
        # print(data)
        if (unique_identifier==None):
            return redirect('/')
        emb_dir_path=os.path.join(BASE_DIR,"static/embeddings/Embeddings.pkl")
        with open(emb_dir_path,"rb") as file:
            emb=pickle.load(file)
        dir=os.path.join(BASE_DIR,f"urlcontent/{unique_identifier}.txt")
        with open(dir,"r") as file:
            content=file.read()
        recursivetextsplitter=RecursiveCharacterTextSplitter(
                separators=['\n\n',"\n","  "," "],
                chunk_size=200,
                chunk_overlap=20,

            )
        chunk_data=recursivetextsplitter.split_text(content)
        
        try:
            embedded_data_path=os.path.join(BASE_DIR,f"static/embedded_data/{unique_identifier}.pkl")
            if os.path.exists(embedded_data_path):
                print("Reading the file >>>>>>")
                with open(embedded_data_path,"rb") as file:
                    embedded_data=pickle.load(file)
                
                    
                    
                    
            else:
                # emb_dir_path=os.path.join(BASE_DIR,"static/embeddings/Embeddings.pkl")
                embedded_data_path=os.path.join(BASE_DIR,f"static/embedded_data/{unique_identifier}.pkl")
                    
                # with open(emb_dir_path,"rb") as file:
                #     emb=pickle.load(file)
                
                embedded_data=emb.embed_documents(chunk_data)
                
                with open(embedded_data_path,"wb") as file:
                    pickle.dump(embedded_data,file)


        
            
            embedded_data=np.array(embedded_data)
            llm=GooglePalm(google_api_key=os.environ['GOOGLE_API_KEY'],
                    temperature=0.5
                )               
            
            index = faiss.IndexFlatL2(768) 
            index.add(embedded_data)   
            query=emb.embed_query(question)
        
            query=np.array([query])
            print("Completeddd 1")
            # print(query)
            dist,index_number=index.search(query,k=10)
           

            another_chunk=""
            for i in range(0,10):
                index_num=index_number[0][i]
                another_chunk+=chunk_data[index_num]
            # print(another_chunk)
            
            template=PromptTemplate(
                input_variables=['question',"databasedata"],
                template="you have to the give the answer based on my question , it is the information taken from the different url {databasedata} it may contain the limitd information , based on the information which was taken from url give the answer based on question , your question is {question} "
            )

            chain=LLMChain(prompt=template,llm=llm)
            
            aimessage=chain.run(question=question,databasedata=another_chunk)
          
        
            return JsonResponse({"message":aimessage})
            
            
        
        except Exception as e:
            print(e)
            return JsonResponse({"message":"uable to find the your url data"})
    
    
     

    
    
    return render(request, "extracor.html")
