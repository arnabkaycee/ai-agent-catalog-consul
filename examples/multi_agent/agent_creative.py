"""
Multi-Agent Example - Creative Agent

This agent generates creative content and narratives.
"""

import os
import sys
import argparse
import json
import consulate  # Python client for Consul

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from a2a.server import run_server


def register_with_consul_kv(consul_client, agent_metadata, agent_name):
    """Register agent metadata with Consul KV."""
    try:
        kv_path = f"a2a/agents/{agent_name}/.well-known/agent.json"
        consul_client.kv.set(kv_path, json.dumps(agent_metadata))
        print(f"Successfully registered agent metadata at {kv_path}")
    except Exception as e:
        print(f"Failed to register agent metadata with Consul KV: {str(e)}")


def register_consul_service(consul_client, agent_name, port, tags=None):
    """Register agent as a service with Consul."""
    if tags is None:
        tags = ["a2a", "agent", "creative"]
        
    try:
        service_definition = {
            "name": agent_name,
            "port": port,
            "tags": tags,
            "check": {
                "http": f"http://localhost:{port}/.well-known/agent.json",
                "interval": "30s"
            }
        }
        
        consul_client.agent.service.register(
            name=agent_name,
            port=port,
            tags=tags
        )
        print(f"Successfully registered service {agent_name} on port {port}")
    except Exception as e:
        print(f"Failed to register service with Consul: {str(e)}")


def main():
    """Run the creative agent server."""
    parser = argparse.ArgumentParser(description="Run Creative Agent")
    parser.add_argument("--model", type=str, default="gemma3:12b-it-qat", help="The Ollama model to use")
    parser.add_argument("--port", type=int, default=8003, help="The port to run the server on")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="The Ollama host URL")
    parser.add_argument("--consul-address", type=str, default="localhost:8500", 
                        help="The address of Consul server (host:port)")
    
    args = parser.parse_args()
    
    # Define the agent's skills
    skills = [
        {
            "id": "content_generation",
            "name": "Content Generation",
            "description": "Generates creative written content on various topics"
        },
        {
            "id": "storytelling",
            "name": "Storytelling",
            "description": "Creates engaging narratives"
        },
        {
            "id": "expression",
            "name": "Expressive Writing",
            "description": "Communicates ideas in an engaging, clear manner"
        }
    ]
    
    # Create a system prompt to guide the model behavior
    system_prompt = """
    You are a specialized Creative Agent that focuses on generating engaging content.
    Your responses should be:
    - Engaging and vivid
    - Well-structured with clear flow
    - Written with an appropriate tone for the subject
    - Concise yet descriptive
    - Designed to evoke interest and emotional connection
    
    As a Creative Agent, your goal is to transform information into compelling narratives that engage readers.
    """
    
    agent_name = "Creative Agent"
    agent_description = "An A2A agent that specializes in generating creative content and narratives"
    
    # Create agent metadata
    agent_metadata = {
        "name": agent_name,
        "description": agent_description,
        "skills": skills,
        "port": args.port
    }
    
    # Setup Consul connection
    consul_host, consul_port = args.consul_address.split(':')
    consul = consulate.Consul(host=consul_host, port=consul_port)
    
    # Register with Consul KV and Service registry
    register_with_consul_kv(consul, agent_metadata, agent_name.lower().replace(" ", "-"))
    register_consul_service(consul, agent_name.lower().replace(" ", "-"), args.port)
    
    # Start the A2A server with the Creative Agent
    run_server(
        model=args.model,
        name=agent_name,
        description=agent_description,
        skills=skills,
        port=args.port,
        ollama_host=args.ollama_host,
        system_prompt=system_prompt
    )


if __name__ == "__main__":
    main()