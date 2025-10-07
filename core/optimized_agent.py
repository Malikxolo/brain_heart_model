"""
Optimized Single-Pass Agent System
Combines semantic analysis, tool execution, and response generation in minimal LLM calls
"""



import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime



logger = logging.getLogger(__name__)


COMPANY_SYSTEM_IDENTITY = """ 
YOU ARE:  
You are the AMi AI assistant — a technical and business-savvy conversational agent representing **AMi**, a high-performance, AI-native cloud infrastructure platform built especially for training, fine-tuning, and serving machine learning models. Your role is to help developers, ML engineers, startups, and enterprises understand, adopt, and scale on AMi in a way that’s efficient, cost-effective, and performance-oriented.

COMPANY CONTEXT:  
- **Company Name:** AMi  
- **Industry:** Cloud infrastructure for AI / GPU-compute platform for machine learning and inference workloads  
- **Core Solutions:**  
  1. **Pod infrastructure** — bare-metal or container-based GPU/CPU compute resources (individual or clustered) for training, development, and experimentation. :contentReference[oaicite:1]{index=1}  
  2. **Serverless endpoints** — auto-scaling, pay-per-use inference infrastructure that handles model serving without managing idle resources. :contentReference[oaicite:2]{index=2}  
  3. **Hub / Templates** — preconfigured open-source repos and templates for common AI workflows, making deployment fast and reproducible. :contentReference[oaicite:3]{index=3}  
  4. **Cluster support** — for distributed training/fine-tuning using GPU clusters with high-bandwidth internode connectivity. :contentReference[oaicite:4]{index=4}  
- **Target Users & Markets:**  
  - Individual developers, researchers, and startups building AI models or prototypes  
  - Product teams deploying LLMs, vision models, or agents into production  
  - Enterprises needing scalable, performant infrastructure without lock-in or high virtualization overhead  
- **Value Proposition & Differentiators:**  
  - High efficiency: near-bare-metal performance by minimizing virtualization overhead, yielding faster training/inference times. :contentReference[oaicite:5]{index=5}  
  - Fast provisioning and autoscaling: spin up compute or endpoints quickly (e.g., sub-200 ms cold starts, always-on workers, autoscale from 0 to many). :contentReference[oaicite:6]{index=6}  
  - Cost savings: usage-based billing, reduced idle costs, and competitive GPU pricing compared to major clouds. :contentReference[oaicite:7]{index=7}  
  - Flexible/developer-friendly: custom Docker containers, CLI/SDK/API access, no lock-in, and template-driven deploys. :contentReference[oaicite:8]{index=8}  
  - Compliance, global reach, and enterprise readiness: security, uptime, regional availability, and scaling. :contentReference[oaicite:9]{index=9}  

AMi-SPECIFIC TRIGGERS:  
Identify these key customer pain points or phrases as signals for deeper engagement or sales qualification:

- **Infrastructure latency / cost inefficiency**  
  - “My cloud GPU is too slow / high latency / virtualization overhead”  
  - “Cold-start time is hurting performance”  
  - “Training/inference costs are too high”  

- **Scaling & reliability challenges**  
  - “We need autoscaling / zero downtime”  
  - “We can’t handle spikes in traffic”  
  - “We need always-on availability or fast cold starts”  

- **Complex deployment operations**  
  - “Managing infrastructure on AWS/GCP is too complex”  
  - “We need custom containers, networking, or orchestration”  

- **Model development speed & iteration challenges**  
  - “We waste too much time setting up clusters or compute”  
  - “Templates / reproducible environments are missing”  

- **Serving & commercialization requirements**  
  - “We need inference endpoints, low latency, high throughput”  
  - “Don’t want to worry about idle compute / overhead costs"  

BUSINESS OPPORTUNITY DETECTION:  
- When users mention **model training, inference, scaling issues, high GPU costs, uptime problems, deployment complexity, or need for fast autoscaling**, treat it as a potential lead.  
- If they compare to AWS/GCP/Azure or express dissatisfaction with current infrastructure, highlight AMi’s performance, cost, and flexibility advantages.  
- Ask qualification questions such as:  
  - “What GPU types and count do you currently use?”  
  - “What’s your usage pattern (continuous training vs intermittent inference)?”  
  - “What throughput and latency targets do you have for serving models?”  
  - “What’s your budget for compute vs existing provider?”  

TOOL USAGE RULES:
- **RAG (Retrieval-Augmented Generation):**  
  - Use AMi’s documentation, pricing pages, benchmarks, case studies, and template repos as the primary source of facts about product features, deployment workflows, and platform capabilities.  
- **WEB_SEARCH:**  
  - Use to compare AMi vs major providers, review third-party benchmarks, market trends in AI infrastructure, and competitor analysis. Do *not* use to override internal specification unless explicitly missing.  
- **CALCULATOR:**  
  - Use for cost/ROI comparisons (e.g., savings vs AWS, pay per use vs fixed cost), scaling analyses, throughput vs infrastructure cost trade-offs.  

COMMUNICATION STYLE:  
- **Tone:** Technical yet accessible. You speak as a trusted infrastructure advisor who understands both ML workflows and business implications.  
- **Style:** Clear, confident, and somewhat consultative. Provide explanations, trade-offs, and recommendations—not just feature lists.  
- **Formality:** Semi-formal, with enough precision for engineers and enough context for business stakeholders.  
- **Personality:** Solution-oriented, practical, performance-driven (“Here’s why AMi can give you faster iterations. Here’s where you should still consider trade-offs.”), empathetic to infrastructure pain.  

OUTPUT REQUIREMENT:  
Your responses should:  
1. Begin by understanding user's goal (training, inference, cost optimization, scaling, etc.).  
2. Provide relevant AMi information, citing documentation (RAG) for features, pricing, compliance, performance.  
3. When appropriate, generate cost or performance estimates using calculator logic and compare to alternatives.  
4. Flag potential upgrades/opportunities (e.g., clustering, templates, serverless endpoints).  
5. Finish with a actionable next step (e.g., “Would you like a demo, pricing quote, or migration plan?”).  
6. Always ask clarifying questions if requirements, budget, runtime, or scale are unclear.  

"""




class OptimizedAgent:
    """Single-pass agent that minimizes LLM calls while maintaining all functionality"""
    
    def __init__(self, brain_llm, heart_llm, tool_manager):
        self.brain_llm = brain_llm
        self.heart_llm = heart_llm
        self.tool_manager = tool_manager
        self.available_tools = tool_manager.get_available_tools()
        logger.info(f"OptimizedAgent initialized with tools: {self.available_tools}")
    
    async def process_query(self, query: str, chat_history: List[Dict] = None, user_id: str = None) -> Dict[str, Any]:
        """Process query with minimal LLM calls"""
        logger.info(f"🚀 PROCESSING QUERY: '{query[:50]}...'")
        start_time = datetime.now()
        
        logger.info(f"🔍 DEBUG CHAT HISTORY:")
        logger.info(f"   Type: {type(chat_history)}")
        logger.info(f"   Length: {len(chat_history) if chat_history else 0}")
        logger.info(f"   Content: {chat_history}")
        logger.info(f"   User ID: {user_id}")
        logger.info(f"   Is None?: {chat_history is None}")
        
        try:
            # STEP 1: Single comprehensive analysis (combines semantic + business + planning)
            analysis_start = datetime.now()
            with open("debug_analysis_prompt.json", "w") as f:
                f.write(json.dumps(chat_history))
            analysis = await self._comprehensive_analysis(query, chat_history)
            analysis_time = (datetime.now() - analysis_start).total_seconds()
            logger.info(f"⏱️ Analysis completed in {analysis_time:.2f}s")
            
            # LOG: Enhanced analysis results
            logger.info(f"📊 ANALYSIS RESULTS:")
            logger.info(f"   Intent: {analysis.get('semantic_intent', 'Unknown')}")
            business_opp = analysis.get('business_opportunity', {})
            logger.info(f"   Business Confidence: {business_opp.get('composite_confidence', 0)}/100")
            logger.info(f"   Engagement Level: {business_opp.get('engagement_level', 'none')}")
            logger.info(f"   Signal Breakdown: {business_opp.get('signal_breakdown', {})}")
            logger.info(f"   Tools Selected: {analysis.get('tools_to_use', [])}")
            logger.info(f"   Response Strategy: {analysis.get('response_strategy', {}).get('personality', 'Unknown')}")
            
            # STEP 2: Execute tools if needed (no LLM calls)
            tool_start = datetime.now()
            tool_results = await self._execute_tools(
                analysis.get('tools_to_use', []),
                query,
                analysis,
                user_id
            )
            tool_time = (datetime.now() - tool_start).total_seconds()
            logger.info(f"⏱️ Tools executed in {tool_time:.2f}s")
            
            
            if tool_results:
                logger.info(f"🛠️ TOOL RESULTS SUMMARY:")
                for tool_name, result in tool_results.items():
                    if isinstance(result, dict) and result.get('success'):
                        logger.info(f"   {tool_name}: SUCCESS - {len(str(result))} chars of data")
                    elif isinstance(result, dict) and 'error' in result:
                        logger.info(f"   {tool_name}: ERROR - {result.get('error', 'Unknown')}")
                    else:
                        logger.info(f"   {tool_name}: RESULT - {type(result)} returned")
            else:
                logger.info(f"🛠️ NO TOOLS EXECUTED - Conversational response only")
            
            
            response_start = datetime.now()
            logger.info(f"💭 PASSING TO RESPONSE GENERATOR:")
            logger.info(f"   Analysis data: {len(str(analysis))} chars")
            logger.info(f"   Tool data: {len(str(tool_results))} chars")
            logger.info(f"   Strategy: {analysis.get('response_strategy', {})}")
            
            final_response = await self._generate_response(
                query,
                analysis,
                tool_results,
                chat_history
            )
            response_time = (datetime.now() - response_start).total_seconds()
            logger.info(f"⏱️ Response generated in {response_time:.2f}s")
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ TOTAL PROCESSING TIME: {total_time:.2f}s (2 LLM calls only)")
            
            return {
                "success": True,
                "response": final_response,
                "analysis": analysis,
                "tool_results": tool_results,
                "tools_used": analysis.get('tools_to_use', []),
                "business_opportunity": analysis.get('business_opportunity', {}),
                "processing_time": {
                    "analysis": analysis_time,
                    "tools": tool_time,
                    "response": response_time,
                    "total": total_time
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error. Please try again."
            }
    def _build_sentiment_language_guide(self, sentiment: Dict) -> str:
        """Build sentiment-driven language guidance"""
        emotion = sentiment.get('primary_emotion', 'casual')
        intensity = sentiment.get('intensity', 'medium')
        
        guides = {
            'frustrated': {
                'high': "User is highly frustrated - use very empathetic, understanding language. Be supportive and understanding", 
                'medium': "User is frustrated - be supportive and understanding",
                'low': "User is mildly frustrated - be gentle and reassuring"
            },
            'excited': {
                'high': "User is very excited - match their energy! Be enthusiastic and positive",
                'medium': "User is excited - be upbeat and encouraging",
                'low': "User is mildly excited - be positive and supportive"
            },
            'confused': {
                'high': "User is very confused - be extra patient and clear. Use simple language",
                'medium': "User is confused - be helpful and explanatory",
                'low': "User is slightly confused - be clarifying but not condescending"
            },
            'urgent': {
                'high': "User needs immediate help - be direct but supportive. Focus on solutions",
                'medium': "User has some urgency - be helpful and action-focused",
                'low': "User has mild urgency - be responsive and solution-oriented"
            },
            'casual': {
                'high': "User is very relaxed - be friendly and conversational",
                'medium': "User is casual - be warm and natural",
                'low': "User is somewhat casual - be friendly but focused"
            }
        }
        
        return guides.get(emotion, {}).get(intensity, "Be naturally helpful and friendly")

    async def _comprehensive_analysis(self, query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Single LLM call for ALL analysis needs"""
        logger.info(f"🔍 ANALYSIS DEBUG:")
        logger.info(f"   Chat History Type: {type(chat_history)}")
        logger.info(f"   Chat History Content: {chat_history}")
        logger.info(f"   Chat History Length: {len(chat_history) if chat_history else 0}")
        
        # Build context from chat history
        context = self._build_context(chat_history)
        logger.info(f"   Built Context: '{context}'")
        
        # Create comprehensive prompt that does everything in one shot
        analysis_prompt = f"""Analyze this query using multi-signal intelligence and return a complete execution plan:

CONVERSATION CONTEXT: {context}
USER QUERY: {query}

AVAILABLE TOOLS:
- web_search: Search internet for current information
- rag: Retrieve from uploaded knowledge base  
- calculator: Perform calculations and analysis

Perform ALL of the following analyses in ONE response:

1. SEMANTIC INTENT (what user really wants)
   - What is the user actually asking for?
   - Does this need current/live information that changes over time?
   - What's their emotional state and communication style?

{COMPANY_SYSTEM_IDENTITY}

Return ONLY valid JSON:
{{
    "semantic_intent": "clear description of what user wants",
    "business_opportunity": {{
        "detected": true/false,
        "composite_confidence": 0-100,
        "engagement_level": "direct_consultation|gentle_suggestion|empathetic_probing|pure_empathy",
        "signal_breakdown": {{
            "work_context": 0-100,
            "emotional_distress": 0-100,
            "solution_seeking": 0-100,
            "scale_scope": 0-100
        }},
        "pain_points": ["specific problems"],
        "solution_areas": ["how we can help"],
        "recommended_approach": "empathy_first|solution_focused|consultation_ready"
    }},
    "tools_to_use": ["tool1", "tool2"],
    "enhanced_queries": {{
        "web_search": "optimized search query",
        "rag": "optimized rag query",
        "calculator": "clear calculation"
    }},
    "tool_reasoning": "why these tools",
    "sentiment": {{
        "primary_emotion": "frustrated|excited|casual|urgent|confused",
        "intensity": "low|medium|high"
    }},
    "response_strategy": {{
        "personality": "empathetic_friend|excited_buddy|helpful_dost|urgent_solver|patient_guide",
        "length": "micro|short|medium|detailed",
        "language": "hinglish|english|professional|casual",
        "tone": "friendly|professional|empathetic|excited"
    }},
    "key_points_to_address": ["point1", "point2"]
}}"""

        try:
            messages = chat_history if chat_history else []
            messages.append({"role": "user", "content": analysis_prompt})
            with open("debug_request_payload.json", "w") as f:
                json.dumps(messages)
            
            logger.info(f"🧠 CALLING BRAIN LLM for analysis...")
            response = await self.brain_llm.generate(
                messages,
                temperature=0.1,
                system_prompt="You are an expert analyst. Analyze queries using multi-signal intelligence covering semantics, business opportunities, tool needs, and communication strategy. Return valid JSON only."
            )
            
            # LOG: Raw LLM response
            logger.info(f"🧠 BRAIN LLM RAW RESPONSE: {len(response)} chars")
            logger.info(f"🧠 First 200 chars: {response[:200]}...")
            
            # Clean response
            cleaned = self._clean_json_response(response)
            logger.info(f"🧠 CLEANED RESPONSE: {len(cleaned)} chars")
            
            analysis = json.loads(cleaned)
            
            # LOG: Parsed analysis details
            logger.info(f"✅ Analysis complete: intent={analysis.get('semantic_intent')}, "
                       f"business={analysis.get('business_opportunity', {}).get('detected')}, "
                       f"confidence={analysis.get('business_opportunity', {}).get('composite_confidence', 0)}, "
                       f"tools={analysis.get('tools_to_use', [])}")
            
            logger.info(f"🧠 FULL ANALYSIS GENERATED:")
            logger.info(f"   Semantic Intent: {analysis.get('semantic_intent', 'Unknown')}")
            logger.info(f"   Business Opportunity: {analysis.get('business_opportunity', {})}")
            logger.info(f"   Tools to Use: {analysis.get('tools_to_use', [])}")
            logger.info(f"   Tool Reasoning: {analysis.get('tool_reasoning', 'None')}")
            logger.info(f"   Sentiment: {analysis.get('sentiment', {})}")
            logger.info(f"   Response Strategy: {analysis.get('response_strategy', {})}")
            logger.info(f"   Enhanced Queries: {analysis.get('enhanced_queries', {})}")
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # Return safe defaults
            return {
                "semantic_intent": "general_inquiry",
                "business_opportunity": {
                    "detected": False, 
                    "composite_confidence": 0,
                    "engagement_level": "pure_empathy",
                    "signal_breakdown": {
                        "work_context": 0,
                        "emotional_distress": 0,
                        "solution_seeking": 0,
                        "scale_scope": 0
                    },
                    "recommended_approach": "empathy_first"
                },
                "tools_to_use": [],
                "enhanced_queries": {},
                "sentiment": {"primary_emotion": "casual", "intensity": "medium"},
                "response_strategy": {
                    "personality": "helpful_dost",
                    "length": "medium",
                    "language": "hinglish",
                    "tone": "friendly"
                }
            }
    
    async def _execute_tools(self, tools: List[str], query: str, analysis: Dict, user_id: str = None) -> Dict[str, Any]:
        """Execute tools with enhanced queries"""
        
        if not tools:
            return {}
        
        results = {}
        enhanced_queries = analysis.get('enhanced_queries', {})
        
        # Execute tools in parallel for speed
        tasks = []
        for tool in tools:
            if tool in self.available_tools:
                # Use enhanced query if available, fallback to raw query
                tool_query = enhanced_queries.get(tool, query)
                logger.info(f"🔧 {tool.upper()} ENHANCED QUERY: '{tool_query}'")
                
                task = self.tool_manager.execute_tool(tool, query=tool_query, user_id=user_id)
                tasks.append((tool, task))
        
        if tasks:
            # Gather all results in parallel
            for tool_name, task in tasks:
                try:
                    result = await task
                    results[tool_name] = result
                    logger.info(f"✅ Tool {tool_name} executed successfully")
                except Exception as e:
                    logger.error(f"❌ Tool {tool_name} failed: {e}")
                    results[tool_name] = {"error": str(e)}
        
        return results

    
    async def _generate_response(self, query: str, analysis: Dict, tool_results: Dict, chat_history: List[Dict]) -> str:
        """Generate response with simple business mode switching like old system"""
        
        # Extract key elements
        intent = analysis.get('semantic_intent', '')
        business_opp = analysis.get('business_opportunity', {})
        sentiment = analysis.get('sentiment', {})
        strategy = analysis.get('response_strategy', {})
        
        # Simple binary business mode logic (like your old system)
        business_detected = business_opp.get('detected', False)
        conversation_mode = "Smart Business Friend" if business_detected else "Casual Dost"
        
        # Actually USE the sentiment guide you built
        sentiment_guidance = self._build_sentiment_language_guide(sentiment)
        
        # Enhanced logging
        logger.info(f"❤️ RESPONSE GENERATION INPUTS:")
        logger.info(f"   Intent: {intent}")
        logger.info(f"   Business Opportunity Detected: {business_detected}")
        logger.info(f"   Conversation Mode: {conversation_mode}")
        logger.info(f"   User Emotion: {sentiment.get('primary_emotion', 'casual')}")
        logger.info(f"   Sentiment Guidance: {sentiment_guidance}")
        logger.info(f"   Response Personality: {strategy.get('personality', 'helpful_dost')}")
        logger.info(f"   Response Length: {strategy.get('length', 'medium')}")
        logger.info(f"   Language Style: {strategy.get('language', 'hinglish')}")
        
        # Format tool results
        tool_data = self._format_tool_results(tool_results)
        logger.info(f"❤️ FORMATTED TOOL DATA: {len(tool_data)} chars")
        
        # Build memory context to avoid repetition
        recent_phrases = self._extract_recent_phrases(chat_history)
        logger.info(f"❤️ RECENT PHRASES TO AVOID: {recent_phrases}")
        
        # Simple, clean prompt (like your old system)
        response_prompt = f"""You are Mochand Dost - a naturally helpful AI friend who becomes a smart business consultant when needed.

    CURRENT CONVERSATION CONTEXT:
    - User Intent: {intent}
    - Business Opportunity Detected: {business_detected}
    - Conversation Mode: {conversation_mode}
    - User Sentiment Guide: {sentiment_guidance}
    - Pain Points: {business_opp.get('pain_points', [])}
    - Solution Areas: {business_opp.get('solution_areas', [])}

    AVAILABLE DATA TO USE NATURALLY:
    {tool_data}

    CONVERSATION MEMORY - Don't repeat these recent patterns:
    {recent_phrases}

    RESPONSE REQUIREMENTS:
    - Personality: {strategy.get('personality', 'helpful_dost')}
    - Length: {strategy.get('length', 'medium')} 
    - Language: {strategy.get('language', 'hinglish')}
    - Tone: {strategy.get('tone', 'friendly')}

    RESPONSE GENERATION RULES:
    1. NEVER restate or echo what the user just said
    2. NEVER announce tool usage ("Let me search...", "I found this data...")  
    3. Use the sentiment guide to match their emotional state perfectly
    4. If business opportunity detected, use progressive disclosure:
    - Early conversations: Focus on understanding their situation through questions
    - After they share pain points and seek guidance: Provide specific Mochand solutions
    - Match their engagement level naturally
    5. Stay in character as their helpful dost friend throughout
    6. End with engaging question to continue conversation naturally
    7. ONLY use information from the available data above - never add facts not provided
    
    Be specific when data is specific:
    ✗ "Some companies..." → ✓ "Apple, Samsung, Xiaomi..."
    ✗ "Market is growing..." → ✓ "Growing 23% CAGR to $46B by 2029..."
    ✗ "Several symptoms..." → ✓ "Fever, lethargy, loss of appetite..."

    USER QUERY: {query}

    Respond naturally as Mochand Dost in {conversation_mode} mode."""

        try:
            # Determine max tokens based on length strategy
            max_tokens = {
                "micro": 150,
                "short": 300,
                "medium": 500,
                "detailed": 700
            }.get(strategy.get('length', 'medium'), 500)
            
            logger.info(f"❤️ CALLING HEART LLM for response generation...")
            logger.info(f"❤️ Max tokens: {max_tokens}, Temperature: 0.4")
            
            messages = chat_history if chat_history else []
            messages.append({"role": "user", "content": response_prompt})
            with open("debug_response_prompt.json", "w") as f:
                f.write(json.dumps(messages))
            response = await self.heart_llm.generate(
                messages,
                temperature=0.4,
                max_tokens=1000,
                system_prompt="You are Mochand Dost, a conversational AI assistant. Always respond with natural, friendly conversation - never with JSON, analysis, or structured data. Be warm and helpful."
            )
            
            # LOG: Raw response from Heart LLM
            logger.info(f"❤️ HEART LLM RAW RESPONSE: {len(response)} chars")
            logger.info(f"❤️ First 200 chars: {response[:200]}...")
            
            # Clean and format
            response = self._clean_response(response)
            logger.info(f"❤️ FINAL CLEANED RESPONSE: {len(response)} chars")
            logger.info(f"❤️ FINAL RESPONSE: {response}")
            
            logger.info(f"✅ Response generated: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I had trouble generating a response. Could you please try again?"
    
    def _build_context(self, chat_history: List[Dict]) -> str:
        """Build context from chat history"""
        if not chat_history:
            return "No previous context"
        
        recent = chat_history
        context_parts = []
        for item in recent:
            if 'query' in item:
                context_parts.append(f"User: {item['query'][:100]}")
            elif 'content' in item:
                context_parts.append(f"{item.get('role', 'user')}: {item['content'][:100]}")
        
        return " \n ".join(context_parts) if context_parts else "No previous context"
    
    def _format_tool_results(self, tool_results: dict) -> str:
        """Format tool results for response generation, handling different tool structures."""
        if not tool_results:
            return "No external data available"
        
        import json
        # Save raw tool results for debugging
        # 👇 ADD THIS ENTIRE BLOCK HERE 👇
        logger.info(f"🔍 RAW TOOL RESULTS DEBUG:")
        for tool_name, result in tool_results.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"TOOL: {tool_name.upper()}")
            logger.info(f"{'='*60}")
            
            if tool_name == 'web_search' and isinstance(result, dict):
                logger.info(f"Web Search Query: {result.get('query', 'N/A')}")
                logger.info(f"Success: {result.get('success', False)}")
                
                if 'results' in result and isinstance(result['results'], list):
                    logger.info(f"Number of results: {len(result['results'])}")
                    
                    for idx, item in enumerate(result['results'][:5]):
                        logger.info(f"\n--- Result {idx+1} ---")
                        logger.info(f"Title: {item.get('title', 'No title')}")
                        logger.info(f"Snippet: {item.get('snippet', 'No snippet')}")
                        logger.info(f"Link: {item.get('link', 'No link')}")
        
        logger.info(f"\n{'='*60}\n")
        
        formatted = []
        
        for tool, result in tool_results.items():
            if isinstance(result, dict) and 'error' not in result:
                # Handle RAG-style result
                if 'success' in result and result['success']:
                    if 'retrieved' in result:
                        retrieved = result.get('retrieved', '')
                        chunks = result.get('chunks', [])
                        for chunk in chunks:
                            logging.info(f"Rag Chunk: {chunk}")
                        formatted.append(f"{tool.upper()} RETRIEVED TEXT:\n{retrieved}\n")
                        if chunks:
                            formatted.append(f"{tool.upper()} CHUNKS:\n" + "\n---\n".join(chunks))
                    
                    # Handle web search-style results
                    elif 'results' in result and isinstance(result['results'], list):
                        formatted.append(f"{tool.upper()} SEARCH RESULTS for query: {result.get('query', '')}\n")
                        for item in result['results']:
                            title = item.get('title', 'No title')
                            snippet = item.get('snippet', '')
                            link = item.get('link', '')
                            formatted.append(f"- {title}\n  {snippet}\n  Link: {link}")
                    
                    # Generic fallback for other data/result keys
                    elif 'data' in result:
                        formatted.append(f"{tool.upper()} DATA:\n{result['data']}")
                    elif 'result' in result:
                        formatted.append(f"{tool.upper()} RESULT:\n{result['result']}")
                    
                    else:
                        formatted.append(f"{tool.upper()}: Success but no recognizable content")
                else:
                    formatted.append(f"{tool.upper()}: No data retrieved or request failed")
            
            elif isinstance(result, str):
                formatted.append(f"{tool.upper()}: {result}")
        
        return "\n\n".join(formatted) if formatted else "No usable tool data"

    def _extract_recent_phrases(self, chat_history: List[Dict]) -> List[str]:
        """Extract recent phrases to avoid repetition"""
        if not chat_history:
            return []
        
        phrases = []
        for msg in chat_history[-4:]:
            if msg.get('role') == 'assistant' or 'response' in msg:
                content = msg.get('content', msg.get('response', ''))
                # Get first 30 chars of each sentence
                sentences = content.split('.')
                for sentence in sentences[:2]:
                    if sentence.strip():
                        phrases.append(sentence.strip()[:30])
        
        return phrases[-5:] if phrases else []
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response for JSON parsing"""
        response = response.strip()
        
        # Remove thinking tags
        if '<think>' in response:
            end_idx = response.find('</think>')
            if end_idx != -1:
                response = response[end_idx + 8:].strip()
        
        # Remove markdown blocks
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean final response for display"""
        # Remove any system thinking
        response = self._clean_json_response(response)
        
        # Fix formatting for display
        response = response.replace('- ', '-')
        response = response.strip()
        
        return response
