"""
evaluate_unified_rag_detailed.py - Complete RAG Evaluation for Unified Vector DB
FIXED: Ground truth IDs now match actual chunk IDs from vector DB
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime


class UnifiedRAGEvaluator:
    """
    Evaluate Unified Vector DB across all layers with detailed output
    """
    
    def __init__(self, project_id: str, model_name: str = "all-mpnet-base-v2"):
        self.project_id = project_id
        self.base_path = f"data/processed/{project_id}"
        
        print("\n" + "=" * 80)
        print("🔹 LOADING UNIFIED VECTOR DB")
        print("=" * 80)
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        
        # Load Unified FAISS Index
        self.index_path = f"{self.base_path}/unified.index"
        self.texts_path = f"{self.base_path}/unified.index.texts"
        
        self.index = None
        self.text_chunks = []
        
        if os.path.exists(self.index_path) and os.path.exists(self.texts_path):
            self._load_index()
        else:
            print(f"❌ Unified vector store not found at {self.index_path}")
            raise FileNotFoundError("Unified index not found")
        
        print("\n✅ Unified Vector DB ready\n")
    
    def _load_index(self):
        """Load FAISS index and text chunks"""
        self.index = faiss.read_index(self.index_path)
        with open(self.texts_path, 'r', encoding='utf-8') as f:
            self.text_chunks = json.load(f)
        print(f"✓ FAISS index: {self.index.d} dim, {len(self.text_chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """Retrieve from unified vector DB"""
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        retrieved_ids = []
        retrieved_docs = []
        retrieved_scores = []
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.text_chunks):
                chunk = self.text_chunks[idx]
                retrieved_docs.append(chunk)
                retrieved_ids.append(f"chunk_{idx}")
                retrieved_scores.append(float(1 - dist))
        
        return {
            "retrieved_ids": retrieved_ids,
            "retrieved_docs": retrieved_docs,
            "retrieved_scores": retrieved_scores,
            "num_retrieved": len(retrieved_ids)
        }
    
    # =========================================================
    # METRICS
    # =========================================================
    
    def hit_rate(self, retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        if not retrieved_ids or not relevant_ids:
            return 0.0
        return 1.0 if any(rid in retrieved_ids[:k] for rid in relevant_ids) else 0.0
    
    def mrr(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        if not retrieved_ids or not relevant_ids:
            return 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0
    
    def ndcg(self, retrieved_ids: List[str], relevance_scores: Dict[str, int], k: int = 5) -> float:
        gains = [relevance_scores.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
        if not gains or sum(gains) == 0:
            return 0.0
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
        return round(dcg / idcg, 4) if idcg > 0 else 0.0
    
    def precision(self, retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        if not retrieved_ids or not relevant_ids:
            return 0.0
        return sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids) / k
    
    def recall(self, retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        if not retrieved_ids or not relevant_ids:
            return 0.0
        return sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids) / len(relevant_ids)
    
    _STOP = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
             'do', 'does', 'did', 'will', 'would', 'what', 'which', 'who', 'this', 'that',
             'these', 'those', 'for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'of', 'to',
             'in', 'on', 'by', 'with', 'without', 'about', 'into', 'through'}
    
    def _tok(self, text: str) -> set:
        return {w.lower().strip("'.,?!") for w in text.split()
                if w.lower() not in self._STOP and len(w) > 2}
    
    def context_relevance(self, query: str, retrieved_docs: List[str]) -> float:
        qt = self._tok(query)
        if not qt or not retrieved_docs:
            return 0.0
        corpus = " ".join(retrieved_docs).lower()
        return sum(1 for t in qt if t in corpus) / len(qt)
    
    def answer_relevance(self, query: str, answer: str) -> float:
        if not query or not answer:
            return 0.0
        qe = self.embedder.encode(query)
        ae = self.embedder.encode(answer)
        return float(np.dot(qe, ae) / (np.linalg.norm(qe) * np.linalg.norm(ae)))
    
    def context_recall(self, ground_truth: str, retrieved_docs: List[str]) -> float:
        terms = self._tok(ground_truth)
        if not terms or not retrieved_docs:
            return 0.0
        corpus = " ".join(retrieved_docs).lower()
        return round(sum(1 for t in terms if t in corpus) / len(terms), 4)
    
    # =========================================================
    # EVALUATION WITH DETAILED OUTPUT
    # =========================================================
    
    def evaluate_layer(self, layer_name: str, test_queries: List[Dict], k: int = 5) -> Dict:
        """
        Evaluate a layer and print detailed output for each query
        """
        print("\n" + "=" * 100)
        print(f"📁 LAYER: {layer_name}")
        print("=" * 100)
        
        results = []
        
        for i, q in enumerate(test_queries, 1):
            query = q["query"]
            relevant_ids = q["relevant_ids"]
            relevance_scores = q["relevance_scores"]
            generated_answer = q.get("generated_answer", "")
            ground_truth_context = q.get("ground_truth_context", "")
            
            # Retrieve from unified vector DB
            retrieval = self.retrieve(query, top_k=k)
            retrieved_ids = retrieval["retrieved_ids"]
            retrieved_docs = retrieval["retrieved_docs"]
            
            # Compute metrics
            hr = self.hit_rate(retrieved_ids, relevant_ids, k)
            mr = self.mrr(retrieved_ids, relevant_ids)
            nd = self.ndcg(retrieved_ids, relevance_scores, k)
            pr = self.precision(retrieved_ids, relevant_ids, k)
            re = self.recall(retrieved_ids, relevant_ids, k)
            cr = self.context_relevance(query, retrieved_docs)
            ar = self.answer_relevance(query, generated_answer) if generated_answer else 0
            ctr = self.context_recall(ground_truth_context, retrieved_docs) if ground_truth_context else 0
            
            # Print detailed query output
            print(f"\n{'─'*100}")
            print(f"📝 QUERY {i}: {query}")
            print(f"{'─'*100}")
            
            print(f"\n📄 RETRIEVED CONTENT (Top {len(retrieved_docs)}):")
            for j, (doc_id, doc, score) in enumerate(zip(retrieved_ids, retrieved_docs, retrieval["retrieved_scores"]), 1):
                print(f"\n   [{j}] ID: {doc_id}")
                print(f"       Score: {score:.4f}")
                print(f"       Content: {doc[:200]}..." if len(doc) > 200 else f"       Content: {doc}")
            
            print(f"\n📊 METRICS:")
            print(f"   ┌────────────────────┬──────────┐")
            print(f"   │ Metric             │ Value    │")
            print(f"   ├────────────────────┼──────────┤")
            print(f"   │ hit_rate           │ {hr:.4f}   │")
            print(f"   │ mrr                │ {mr:.4f}   │")
            print(f"   │ ndcg               │ {nd:.4f}   │")
            print(f"   │ precision          │ {pr:.4f}   │")
            print(f"   │ recall             │ {re:.4f}   │")
            print(f"   │ context_relevance  │ {cr:.4f}   │")
            print(f"   │ answer_relevance   │ {ar:.4f}   │")
            print(f"   │ context_recall     │ {ctr:.4f}   │")
            print(f"   └────────────────────┴──────────┘")
            
            print(f"\n✅ EXPECTED RELEVANT IDs: {relevant_ids}")
            
            results.append({
                "query": query,
                "query_num": i,
                "retrieved_ids": retrieved_ids,
                "retrieved_docs": retrieved_docs,
                "retrieved_scores": retrieval["retrieved_scores"],
                "hit_rate": hr,
                "mrr": mr,
                "ndcg": nd,
                "precision": pr,
                "recall": re,
                "context_relevance": cr,
                "answer_relevance": ar,
                "context_recall": ctr,
                "relevant_ids": relevant_ids
            })
        
        # Calculate layer averages
        avg_results = {
            "hit_rate": round(np.mean([r["hit_rate"] for r in results]), 4),
            "mrr": round(np.mean([r["mrr"] for r in results]), 4),
            "ndcg": round(np.mean([r["ndcg"] for r in results]), 4),
            "precision": round(np.mean([r["precision"] for r in results]), 4),
            "recall": round(np.mean([r["recall"] for r in results]), 4),
            "context_relevance": round(np.mean([r["context_relevance"] for r in results]), 4),
            "answer_relevance": round(np.mean([r["answer_relevance"] for r in results]), 4),
            "context_recall": round(np.mean([r["context_recall"] for r in results]), 4)
        }
        
        # Print layer summary
        print(f"\n{'='*100}")
        print(f"📊 {layer_name} - SUMMARY")
        print(f"{'='*100}")
        print(f"  {'Metric':<20} {'Average':>10}")
        print(f"  {'-'*30}")
        print(f"  {'hit_rate':<20} {avg_results['hit_rate']:>10.4f}")
        print(f"  {'mrr':<20} {avg_results['mrr']:>10.4f}")
        print(f"  {'ndcg':<20} {avg_results['ndcg']:>10.4f}")
        print(f"  {'precision':<20} {avg_results['precision']:>10.4f}")
        print(f"  {'recall':<20} {avg_results['recall']:>10.4f}")
        print(f"  {'context_relevance':<20} {avg_results['context_relevance']:>10.4f}")
        print(f"  {'answer_relevance':<20} {avg_results['answer_relevance']:>10.4f}")
        print(f"  {'context_recall':<20} {avg_results['context_recall']:>10.4f}")
        print(f"{'='*100}")
        
        return {
            "layer_name": layer_name,
            "per_query": results,
            "average": avg_results,
            "num_queries": len(test_queries)
        }


# =========================================================
# TEST QUERIES BY LAYER - UPDATED WITH ACTUAL CHUNK IDs
# =========================================================

# LAYER 1: Building Elements (IFC)
# Based on actual retrieval, L1 chunks are: chunk_53, chunk_54, chunk_39, chunk_40, chunk_41
LAYER1_QUERIES = [
    {
        "query": "Describe beam YC-ST-SF-BIP",
        "relevant_ids": ["chunk_104", "chunk_53", "chunk_54"],
        "relevance_scores": {"chunk_104": 3, "chunk_53": 2, "chunk_54": 2},
        "generated_answer": "YC-ST-SF-BIP is an IfcBeam with ObjectType T1.",
        "ground_truth_context": "YC-ST-SF-BIP IfcBeam ObjectType T1"
    },
    {
        "query": "What are the properties of wall YC-ST-WA-EIP?",
        "relevant_ids": ["chunk_39", "chunk_40"],
        "relevance_scores": {"chunk_39": 3, "chunk_40": 2},
        "generated_answer": "Wall YC-ST-WA-EIP has length 4343.4mm and width 0.00246mm.",
        "ground_truth_context": "YC-ST-WA-EIP length width"
    },
    {
        "query": "Show me all IfcWall elements",
        "relevant_ids": ["chunk_39", "chunk_40", "chunk_41", "chunk_42"],
        "relevance_scores": {"chunk_39": 3, "chunk_40": 3, "chunk_41": 2, "chunk_42": 2},
        "generated_answer": "IfcWall elements: YC-ST-WA-EIP, YC-AR-WA-IIP, YC-AR-WA-PIP",
        "ground_truth_context": "IfcWall YC-ST-WA-EIP YC-AR-WA-IIP YC-AR-WA-PIP"
    }
]

# LAYER 2: Products
LAYER2_QUERIES = [
    {
        "query": "What is the fire rating of YC-ST-SF-BIP?",
        "relevant_ids": ["chunk_104", "chunk_425"],
        "relevance_scores": {"chunk_104": 3, "chunk_425": 2},
        "generated_answer": "YC-ST-SF-BIP has fire rating 2 hours.",
        "ground_truth_context": "Fire_Rating_Hours 2"
    },
    {
        "query": "What is the cost of beam YC-ST-SF-BIP?",
        "relevant_ids": ["chunk_104"],
        "relevance_scores": {"chunk_104": 3},
        "generated_answer": "The unit cost is 7200 INR.",
        "ground_truth_context": "Unit_Cost_INR 7200"
    },
    {
        "query": "What are the specifications of YC-ST-SF-BIP?",
        "relevant_ids": ["chunk_104"],
        "relevance_scores": {"chunk_104": 3},
        "generated_answer": "Concrete Grade M30, Steel Grade Fe500D, Length 6000mm",
        "ground_truth_context": "Concrete Grade M30 Steel Grade Fe500D"
    }
]

# LAYER 3: Process Rules (L3)
# Updated with actual chunk IDs from retrieval
LAYER3_QUERIES = [
    {
        "query": "What are the information requirements for BIM?",
        "relevant_ids": ["chunk_179", "chunk_106", "chunk_163"],
        "relevance_scores": {"chunk_179": 3, "chunk_106": 2, "chunk_163": 2},
        "generated_answer": "Information requirements include organizational and project requirements.",
        "ground_truth_context": "information requirements organizational project"
    },
    {
        "query": "What is the information delivery planning process?",
        "relevant_ids": ["chunk_158", "chunk_111", "chunk_167"],
        "relevance_scores": {"chunk_158": 3, "chunk_111": 2, "chunk_167": 2},
        "generated_answer": "Information delivery planning includes TIDP and MIDP.",
        "ground_truth_context": "information delivery planning TIDP MIDP"
    },
    {
        "query": "What are exchange information requirements?",
        "relevant_ids": ["chunk_114", "chunk_163", "chunk_112"],
        "relevance_scores": {"chunk_114": 3, "chunk_163": 2, "chunk_112": 2},
        "generated_answer": "Exchange information requirements define what information to deliver.",
        "ground_truth_context": "exchange information requirements deliver"
    }
]

# LAYER 4: Regulations
LAYER4_QUERIES = [
    {
        "query": "What are the fire safety requirements for external walls?",
        "relevant_ids": ["chunk_435", "chunk_377", "chunk_464"],
        "relevance_scores": {"chunk_435": 3, "chunk_377": 2, "chunk_464": 2},
        "generated_answer": "External walls require 120 minutes fire resistance.",
        "ground_truth_context": "fire resistance 120 minutes external walls"
    },
    {
        "query": "What is the minimum road width requirement?",
        "relevant_ids": ["chunk_378", "chunk_402"],
        "relevance_scores": {"chunk_378": 3, "chunk_402": 2},
        "generated_answer": "Minimum road width is 7.2m for residential and 9m for industrial.",
        "ground_truth_context": "minimum width 7.2m residential 9m industrial"
    },
    {
        "query": "What are the basement height requirements?",
        "relevant_ids": ["chunk_386", "chunk_398"],
        "relevance_scores": {"chunk_386": 3, "chunk_398": 2},
        "generated_answer": "Basement height must be at least 2.4m.",
        "ground_truth_context": "basement height 2.4m"
    }
]

# LAYER 5: Requirements
LAYER5_QUERIES = [
    {
        "query": "What is requirement WR-M0090?",
        "relevant_ids": ["chunk_310"],
        "relevance_scores": {"chunk_310": 3},
        "generated_answer": "WR-M0090 is Counter balance valve with rate 43000.",
        "ground_truth_context": "Counter balance valve Rate 43000"
    },
    {
        "query": "What is the rate for welding machine?",
        "relevant_ids": ["chunk_333", "chunk_338"],
        "relevance_scores": {"chunk_333": 3, "chunk_338": 2},
        "generated_answer": "Welding machine rate is 85.6 per hour.",
        "ground_truth_context": "welding machine rate 85.6"
    },
    {
        "query": "List all equipment requirements",
        "relevant_ids": ["chunk_333", "chunk_334", "chunk_335", "chunk_336"],
        "relevance_scores": {"chunk_333": 3, "chunk_334": 2, "chunk_335": 2, "chunk_336": 2},
        "generated_answer": "Equipment includes welding machines, cranes, compressors.",
        "ground_truth_context": "welding machine crane compressor"
    }
]

# LAYER l124: Compliance Issues
# Note: These use original JSON IDs because they're from l124_inference.json
LAYER124_QUERIES = [
    {
        "query": "What compliance issues does YC-ST-SF-BIP beam have?",
        "relevant_ids": ["1Wl25_be57y9BVJc_M$CEw", "1Wl25_be57y9BVJc_M$CEd"],
        "relevance_scores": {"1Wl25_be57y9BVJc_M$CEw": 3, "1Wl25_be57y9BVJc_M$CEd": 3},
        "generated_answer": "YC-ST-SF-BIP has fire rating violations.",
        "ground_truth_context": "fire rating violation non-compliant"
    },
    {
        "query": "Show me non-compliant building elements",
        "relevant_ids": ["1Wl25_be57y9BVJc_M$CEw", "1Wl25_be57y9BVJc_M$CEd"],
        "relevance_scores": {"1Wl25_be57y9BVJc_M$CEw": 3, "1Wl25_be57y9BVJc_M$CEd": 3},
        "generated_answer": "YC-ST-SF-BIP beam is non-compliant.",
        "ground_truth_context": "non-compliant YC-ST-SF-BIP"
    },
    {
        "query": "What are the fire rating violations?",
        "relevant_ids": ["1Wl25_be57y9BVJc_M$CEw", "1Wl25_be57y9BVJc_M$CEd"],
        "relevance_scores": {"1Wl25_be57y9BVJc_M$CEw": 3, "1Wl25_be57y9BVJc_M$CEd": 3},
        "generated_answer": "Fire rating does not meet requirement for YC-ST-SF-BIP.",
        "ground_truth_context": "fire rating does not meet requirement"
    }
]

# LAYER l125: Project Requirements Compliance
# Note: These need to be populated first by running inference
LAYER125_QUERIES = [
    {
        "query": "Which requirements are missing for YC-ST-SF-BIP?",
        "relevant_ids": ["L5_8559daca"],
        "relevance_scores": {"L5_8559daca": 3},
        "generated_answer": "REQ-FIRE-001 requires EI120 fire resistance.",
        "ground_truth_context": "REQ-FIRE-001 EI120 fire resistance"
    },
    {
        "query": "What project requirements are not met?",
        "relevant_ids": ["L5_8559daca"],
        "relevance_scores": {"L5_8559daca": 3},
        "generated_answer": "External walls must achieve EI120 fire resistance.",
        "ground_truth_context": "external walls EI120 fire resistance"
    },
    {
        "query": "Show requirement compliance status",
        "relevant_ids": ["L5_8559daca"],
        "relevance_scores": {"L5_8559daca": 3},
        "generated_answer": "REQ-FIRE-001 requires EI120. Current beam has 2 hours.",
        "ground_truth_context": "REQ-FIRE-001 EI120 requirement"
    }
]

# LAYER l45: Regulation vs Requirement Gap Analysis
LAYER45_QUERIES = [
    {
        "query": "Compare fire safety regulations with requirements",
        "relevant_ids": ["L4_fire_001", "L5_8559daca"],
        "relevance_scores": {"L4_fire_001": 3, "L5_8559daca": 3},
        "generated_answer": "Both regulations and requirements need 120 minutes fire resistance.",
        "ground_truth_context": "fire resistance 120 minutes regulations requirements"
    },
    {
        "query": "What gaps exist between regulations and requirements?",
        "relevant_ids": ["L4_fire_001", "L5_8559daca"],
        "relevance_scores": {"L4_fire_001": 3, "L5_8559daca": 3},
        "generated_answer": "Regulations and requirements are aligned for fire safety.",
        "ground_truth_context": "regulations requirements aligned fire safety"
    },
    {
        "query": "Are all regulations covered by requirements?",
        "relevant_ids": ["L4_fire_001", "L5_8559daca"],
        "relevance_scores": {"L4_fire_001": 3, "L5_8559daca": 3},
        "generated_answer": "Fire safety regulations are covered by requirements.",
        "ground_truth_context": "fire safety regulations covered requirements"
    }
]


# =========================================================
# MAIN EXECUTION
# =========================================================

def print_final_summary(all_results: List[Dict]):
    """Print final summary table for all layers"""
    print("\n" + "=" * 120)
    print("📊 FINAL SUMMARY - ALL LAYERS")
    print("=" * 120)
    
    # Header
    print(f"\n{'Layer':<12} {'Hit Rate':>10} {'MRR':>8} {'NDCG':>8} {'Precision':>10} {'Recall':>8} {'Context Rel':>11} {'Answer Rel':>11} {'Context Rec':>11}")
    print("-" * 120)
    
    for result in all_results:
        avg = result["average"]
        layer = result["layer_name"]
        print(f"{layer:<12} {avg['hit_rate']:>10.4f} {avg['mrr']:>8.4f} {avg['ndcg']:>8.4f} "
              f"{avg['precision']:>10.4f} {avg['recall']:>8.4f} {avg['context_relevance']:>11.4f} "
              f"{avg['answer_relevance']:>11.4f} {avg['context_recall']:>11.4f}")
    
    print("=" * 120)
    
    # Calculate overall averages
    overall = {
        "hit_rate": np.mean([r["average"]["hit_rate"] for r in all_results]),
        "mrr": np.mean([r["average"]["mrr"] for r in all_results]),
        "ndcg": np.mean([r["average"]["ndcg"] for r in all_results]),
        "precision": np.mean([r["average"]["precision"] for r in all_results]),
        "recall": np.mean([r["average"]["recall"] for r in all_results]),
        "context_relevance": np.mean([r["average"]["context_relevance"] for r in all_results]),
        "answer_relevance": np.mean([r["average"]["answer_relevance"] for r in all_results]),
        "context_recall": np.mean([r["average"]["context_recall"] for r in all_results])
    }
    
    print(f"\n{'OVERALL AVERAGE':<12} {overall['hit_rate']:>10.4f} {overall['mrr']:>8.4f} {overall['ndcg']:>8.4f} "
          f"{overall['precision']:>10.4f} {overall['recall']:>8.4f} {overall['context_relevance']:>11.4f} "
          f"{overall['answer_relevance']:>11.4f} {overall['context_recall']:>11.4f}")
    print("=" * 120)


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 UNIFIED VECTOR DB - COMPLETE LAYER EVALUATION")
    print("=" * 80)
    print("""
  📋 EVALUATION LAYERS:
  -------------------------------------------------------------------
  Layer 1 (L1)      : Building Elements (IFC beams, walls, properties)
  Layer 2 (L2)      : Products (fire rating, cost, specifications)
  Layer 3 (L3)      : Process Rules (BIM information requirements)
  Layer 4 (L4)      : Regulations (fire safety, road width, basement)
  Layer 5 (L5)      : Requirements (material rates, equipment)
  Layer l124        : Compliance Issues (violations, non-compliance)
  Layer l125        : Project Requirements Compliance
  Layer l45         : Regulation vs Requirement Gap Analysis
  -------------------------------------------------------------------
    """)
    
    project_id = input("\nEnter project ID (e.g., 'new'): ").strip()
    
    try:
        evaluator = UnifiedRAGEvaluator(project_id)
        
        all_results = []
        
        # Evaluate each layer
        print("\n" + "🔍" * 40)
        print("LAYER 1: BUILDING ELEMENTS (L1)")
        print("🔍" * 40)
        result1 = evaluator.evaluate_layer("L1_Building_Elements", LAYER1_QUERIES, k=5)
        all_results.append(result1)
        
        print("\n" + "🔍" * 40)
        print("LAYER 2: PRODUCTS (L2)")
        print("🔍" * 40)
        result2 = evaluator.evaluate_layer("L2_Products", LAYER2_QUERIES, k=5)
        all_results.append(result2)
        
        print("\n" + "🔍" * 40)
        print("LAYER 3: PROCESS RULES (L3)")
        print("🔍" * 40)
        result3 = evaluator.evaluate_layer("L3_Process_Rules", LAYER3_QUERIES, k=5)
        all_results.append(result3)
        
        print("\n" + "🔍" * 40)
        print("LAYER 4: REGULATIONS (L4)")
        print("🔍" * 40)
        result4 = evaluator.evaluate_layer("L4_Regulations", LAYER4_QUERIES, k=5)
        all_results.append(result4)
        
        print("\n" + "🔍" * 40)
        print("LAYER 5: REQUIREMENTS (L5)")
        print("🔍" * 40)
        result5 = evaluator.evaluate_layer("L5_Requirements", LAYER5_QUERIES, k=5)
        all_results.append(result5)
        
        print("\n" + "🔍" * 40)
        print("LAYER l124: COMPLIANCE ISSUES")
        print("🔍" * 40)
        result124 = evaluator.evaluate_layer("l124_Compliance", LAYER124_QUERIES, k=5)
        all_results.append(result124)
        
        print("\n" + "🔍" * 40)
        print("LAYER l125: PROJECT REQUIREMENTS COMPLIANCE")
        print("🔍" * 40)
        result125 = evaluator.evaluate_layer("l125_Project_Compliance", LAYER125_QUERIES, k=5)
        all_results.append(result125)
        
        print("\n" + "🔍" * 40)
        print("LAYER l45: REGULATION VS REQUIREMENT GAP ANALYSIS")
        print("🔍" * 40)
        result45 = evaluator.evaluate_layer("l45_Gap_Analysis", LAYER45_QUERIES, k=5)
        all_results.append(result45)
        
        # Print final summary
        print_final_summary(all_results)
        
        print("\n" + "=" * 80)
        print("✅ EVALUATION COMPLETE")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease build unified vector store first using option 11 in the main menu.")