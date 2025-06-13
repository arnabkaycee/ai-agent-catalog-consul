"""
Multi-Agent Example - Orchestrator

This module coordinates multiple specialized A2A agents to complete complex tasks.
"""

import os
import sys
import argparse
import time
import json
import consulate
import requests
from typing import Dict, Any, List, Optional

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from a2a.client import A2AClient


class AgentOrchestrator:
    """
    Orchestrator for coordinating multiple A2A agents.
    """
    
    def __init__(
        self,
        consul_address: str = "localhost:8500",
        ollama_model: str = "gemma3:12b-it-qat",
        ollama_host: str = "http://localhost:11434"
    ):
        """
        Initialize the orchestrator.
        
        Args:
            consul_address: Address of the Consul server
            ollama_model: The Ollama model to use for delegation decisions
            ollama_host: Endpoint for the Ollama API
        """
        self.consul_address = consul_address
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        
        # Set up Consul client
        consul_host, consul_port = consul_address.split(':')
        self.consul = consulate.Consul(host=consul_host, port=consul_port)
        
        # Dictionary to cache agent clients
        self.agent_clients = {}
        
        # Discover available agents
        self.available_agents = self.discover_agents()
        if not self.available_agents:
            print("Error: No A2A agents found in Consul registry.")
            sys.exit(1)
            
        print(f"Discovered {len(self.available_agents)} agents in Consul registry.")
        for agent in self.available_agents:
            print(f"✓ Found {agent['name']} at http://{agent['address']}:{agent['port']}")
            print(f"  Skills: {', '.join(skill['name'] for skill in agent['skills'])}")
            
        print("\nAll agents discovered successfully.\n")
    
    def discover_agents(self) -> List[Dict[str, Any]]:
        """
        Discover available A2A agents from Consul.
        
        Returns:
            List of agent metadata
        """
        agents = []
        
        try:
            # Get all services with tag "a2a" and "agent"
            services = self.consul.agent.services()
            
            # Debug the services response
            print(f"Service response type: {type(services)}")
            print(f"Service response content: {services}")
            
            # For consulate API, we need to look if it's returning a string and try to parse it
            if isinstance(services, str):
                try:
                    # Try to parse it as JSON
                    services = json.loads(services)
                    print("Successfully parsed service string as JSON")
                except json.JSONDecodeError:
                    print("Could not parse service string as JSON")
                    
                    # Try to extract service IDs from raw string response
                    # This is a fallback approach if we can't parse the JSON
                    # Direct access to service registry
                    try:
                        print("Trying direct API call to Consul")
                        consul_host, consul_port = self.consul_address.split(':')
                        response = requests.get(f"http://{consul_host}:{consul_port}/v1/agent/services")
                        if response.status_code == 200:
                            services = response.json()
                            print("Successfully retrieved services using direct API call")
                        else:
                            print(f"Failed to retrieve services: {response.status_code}")
                            return agents
                    except Exception as e:
                        print(f"Error in direct API call: {e}")
                        return agents
            
            # Handle different response formats from Consul
            if isinstance(services, dict):
                service_items = list(services.items())
                print(f"Found {len(service_items)} services in dictionary format")
            elif hasattr(services, 'items') and callable(services.items):
                service_items = list(services.items())
                print(f"Found {len(service_items)} services using items() method")
            elif isinstance(services, list):
                # Handle list of services
                service_items = [(service.get("ID", ""), service) for service in services if isinstance(service, dict)]
                print(f"Found {len(service_items)} services in list format")
            else:
                print(f"Unexpected services response format: {type(services)}")
                # Try to convert to a list if it's another iterable
                try:
                    service_items = [(str(i), service) for i, service in enumerate(services)]
                    print(f"Converted to {len(service_items)} service items")
                except:
                    print("Could not process services response")
                    return agents
            
            # Print first service for debugging
            if service_items:
                print(f"First service: {service_items[0]}")
            
            for service_id, service in service_items:
                if service is None:
                    continue
                    
                # Make sure service is a dictionary
                if not isinstance(service, dict):
                    print(f"Service is not a dictionary: {type(service)}")
                    
                    # Try to convert string to dictionary if possible
                    if isinstance(service, str):
                        try:
                            service = json.loads(service)
                            print("Successfully parsed service string as JSON")
                        except json.JSONDecodeError:
                            print(f"Could not parse service as JSON: {service[:100]}")
                            continue
                    else:
                        continue
                    
                # Get tags, handling different possible formats
                tags = service.get("Tags", [])
                if isinstance(tags, str):
                    tags = [tags]
                elif not isinstance(tags, list):
                    try:
                        tags = list(tags)
                    except:
                        tags = []
                
                print(f"Service {service_id} tags: {tags}")
                
                if "a2a" in tags and "agent" in tags:
                    # Get agent metadata from KV store
                    service_name = service.get("Service", service_id)
                    kv_path = f"a2a/agents/{service_name}/.well-known/agent.json"
                    
                    try:
                        print(f"Looking for metadata at KV path: {kv_path}")
                        metadata_json = self.consul.kv.get(kv_path)
                        
                        if metadata_json:
                            print(f"Found metadata: {metadata_json[:100]}")
                            
                            # Handle Consul KV response formats
                            try:
                                # Check if it's a string that needs to be parsed as JSON
                                if isinstance(metadata_json, str):
                                    # Try to parse as a JSON array first
                                    try:
                                        parsed_json = json.loads(metadata_json)
                                        # If it's an array with KV objects
                                        if isinstance(parsed_json, list) and len(parsed_json) > 0 and "Value" in parsed_json[0]:
                                            # Extract the Base64 encoded value
                                            import base64
                                            value_b64 = parsed_json[0].get("Value", "")
                                            if value_b64:
                                                # Decode base64 value
                                                decoded_value = base64.b64decode(value_b64).decode('utf-8')
                                                metadata = json.loads(decoded_value)
                                            else:
                                                print(f"No Value field found in KV response")
                                                continue
                                        else:
                                            # It's already parsed JSON
                                            metadata = parsed_json
                                    except json.JSONDecodeError:
                                        # Not valid JSON, use as is
                                        print(f"Could not parse metadata as JSON, using raw value")
                                        metadata = {"name": service_name, "description": "", "skills": []}
                                else:
                                    # Not a string, use as is
                                    metadata = metadata_json
                                    
                                # Add service information
                                metadata["address"] = service.get("Address", "") or "localhost"
                                metadata["port"] = service.get("Port", 8000)
                                agents.append(metadata)
                                print(f"Added agent: {metadata.get('name', service_name)}")
                            except Exception as e:
                                print(f"Error processing metadata: {str(e)}")
                                
                                # Try direct KV API as fallback with raw=true parameter
                                try:
                                    consul_host, consul_port = self.consul_address.split(':')
                                    kv_url = f"http://{consul_host}:{consul_port}/v1/kv/{kv_path}?raw=true"
                                    response = requests.get(kv_url)
                                    
                                    if response.status_code == 200:
                                        raw_metadata = response.text
                                        try:
                                            metadata = json.loads(raw_metadata)
                                            metadata["address"] = service.get("Address", "") or "localhost"
                                            metadata["port"] = service.get("Port", 8000)
                                            agents.append(metadata)
                                            print(f"Added agent via direct KV raw API: {metadata.get('name', service_name)}")
                                        except json.JSONDecodeError:
                                            print(f"Could not parse raw metadata response as JSON")
                                    else:
                                        print(f"KV direct API call failed: {response.status_code}")
                                except Exception as fallback_error:
                                    print(f"Error in direct KV API fallback: {str(fallback_error)}")
                        else:
                            print(f"No metadata found at {kv_path}")
                            
                            # Try direct KV API as fallback - use both regular and raw formats
                            try:
                                consul_host, consul_port = self.consul_address.split(':')
                                
                                # First try with raw=true to get the direct value
                                kv_url_raw = f"http://{consul_host}:{consul_port}/v1/kv/{kv_path}?raw=true"
                                response_raw = requests.get(kv_url_raw)
                                
                                if response_raw.status_code == 200:
                                    try:
                                        metadata = json.loads(response_raw.text)
                                        metadata["address"] = service.get("Address", "") or "localhost"
                                        metadata["port"] = service.get("Port", 8000)
                                        agents.append(metadata)
                                        print(f"Added agent via direct raw KV API: {metadata.get('name', service_name)}")
                                        continue
                                    except json.JSONDecodeError:
                                        print("Could not parse raw response as JSON")
                                
                                # If raw failed, try regular format to decode base64
                                kv_url = f"http://{consul_host}:{consul_port}/v1/kv/{kv_path}"
                                response = requests.get(kv_url)
                                
                                if response.status_code == 200:
                                    try:
                                        kv_data = response.json()
                                        if isinstance(kv_data, list) and len(kv_data) > 0 and "Value" in kv_data[0]:
                                            # Extract Base64 encoded value
                                            import base64
                                            value_b64 = kv_data[0].get("Value", "")
                                            if value_b64:
                                                # Decode base64 value
                                                decoded_value = base64.b64decode(value_b64).decode('utf-8')
                                                metadata = json.loads(decoded_value)
                                                metadata["address"] = service.get("Address", "") or "localhost"
                                                metadata["port"] = service.get("Port", 8000)
                                                agents.append(metadata)
                                                print(f"Added agent via base64 decode: {metadata.get('name', service_name)}")
                                    except Exception as e:
                                        print(f"Error processing KV data: {str(e)}")
                                else:
                                    print(f"KV direct API call failed: {response.status_code}")
                                    
                                # As a final fallback, create a basic agent from service info
                                if not any(a.get("name", "") == service_name for a in agents):
                                    basic_agent = {
                                        "name": service_name,
                                        "description": f"Agent for {service_name}",
                                        "skills": [{"name": tag, "description": f"{tag} capability", "id": tag} 
                                                for tag in service.get("Tags", []) if tag not in ["a2a", "agent"]],
                                        "address": service.get("Address", "") or "localhost",
                                        "port": service.get("Port", 8000)
                                    }
                                    agents.append(basic_agent)
                                    print(f"Added basic agent from service info: {service_name}")
                            except Exception as e:
                                print(f"Error in direct KV API call: {e}")
                    except Exception as e:
                        print(f"Warning: Could not retrieve metadata for {service_name}: {e}")
                        
        except Exception as e:
            print(f"Error discovering agents from Consul: {str(e)}")
            import traceback
            traceback.print_exc()
            
        print(f"Total agents found: {len(agents)}")
        return agents
    
    def get_agent_for_task(self, task: str) -> Dict[str, Any]:
        """
        Use Ollama to decide which agent is best suited for a given task.
        
        Args:
            task: Description of the task
            
        Returns:
            Selected agent metadata
        """
        # Create a prompt for Ollama to determine the best agent
        agent_descriptions = []
        for i, agent in enumerate(self.available_agents):
            skills_text = ", ".join([f"{skill['name']}: {skill['description']}" for skill in agent.get("skills", [])])
            agent_descriptions.append(f"Agent {i+1}: {agent['name']} - {agent.get('description', '')} - Skills: {skills_text}")
            
        prompt = f"""
        Based on the following task, select the most appropriate agent by returning ONLY the number of the agent.
        
        TASK: {task}
        
        AVAILABLE AGENTS:
        {chr(10).join(agent_descriptions)}
        
        Return ONLY the number (e.g., "1", "2", "3") of the most appropriate agent for this task. Do not include any explanation or other text.
        """
        
        try:
            # Make a request to Ollama
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                agent_number_text = result.get("response", "").strip()
                
                # Extract just the number from the response
                import re
                match = re.search(r'\d+', agent_number_text)
                if match:
                    agent_number = int(match.group()) - 1
                    if 0 <= agent_number < len(self.available_agents):
                        selected_agent = self.available_agents[agent_number]
                        print(f"Selected {selected_agent['name']} for task: {task[:50]}...")
                        return selected_agent
            
            # Default to the first agent if parsing fails
            print(f"Could not determine appropriate agent. Using {self.available_agents[0]['name']} as default.")
            return self.available_agents[0]
            
        except Exception as e:
            print(f"Error selecting agent: {e}")
            return self.available_agents[0]  # Default to first agent on error
    
    def get_agent_client(self, agent_metadata: Dict[str, Any]) -> A2AClient:
        """
        Get or create an A2A client for the specified agent.
        
        Args:
            agent_metadata: The agent metadata from Consul
            
        Returns:
            A2AClient for the agent
        """
        agent_name = agent_metadata["name"]
        
        if agent_name not in self.agent_clients:
            endpoint = f"http://{agent_metadata['address']}:{agent_metadata['port']}"
            self.agent_clients[agent_name] = A2AClient(endpoint)
            
        return self.agent_clients[agent_name]
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract text content from an agent response."""
        if "message" in response:
            for part in response["message"]["parts"]:
                if part["type"] == "text":
                    return part["content"]
        
        return str(response)  # Fallback
    
    def process_request(self, request: str) -> Dict[str, str]:
        """
        Process a request using a single appropriately selected agent.
        
        Args:
            request: The user request/prompt
            
        Returns:
            Dictionary with the agent's response and metadata
        """
        print(f"Processing request: '{request[:100]}...' if len(request) > 100 else request")
        
        # Select appropriate agent for the task
        selected_agent = self.get_agent_for_task(request)
        agent_client = self.get_agent_client(selected_agent)
        
        # Send request to the selected agent
        print(f"\nDelegating request to {selected_agent['name']}...")
        agent_response = agent_client.chat(request)
        response_content = self._extract_content(agent_response)
        
        # Return the response with metadata
        return {
            "content": response_content,
            "agent": selected_agent['name'],
            "agent_metadata": selected_agent
        }


def main():
    """Run the multi-agent orchestration example."""
    parser = argparse.ArgumentParser(description="A2A Multi-Agent Orchestrator")
    parser.add_argument("--request", type=str, required=True, 
                        help="The request to process (include keywords like 'research', 'analyze', or 'create')")
    parser.add_argument("--consul-address", type=str, default="localhost:8500", 
                        help="Consul server address")
    parser.add_argument("--ollama-model", type=str, default="gemma3:12b-it-qat", 
                        help="Ollama model for delegation decisions")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", 
                        help="Ollama API endpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional filename to save the response")
    
    args = parser.parse_args()
    
    # Create the orchestrator
    orchestrator = AgentOrchestrator(
        consul_address=args.consul_address,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host
    )
    
    # Process the request with the appropriate agent
    response = orchestrator.process_request(args.request)
    
    print("\n" + "="*80)
    print(f"\nRESPONSE FROM {response['agent']}:\n")
    print(response['content'])
    print("\n" + "="*80)
    
    # Save the response to a file if requested
    if args.output:
        filename = args.output
    else:
        # Generate a default filename based on the first few words of the request
        words = args.request.split()[:5]
        safe_words = [word.replace('/', '_').replace('\\', '_') for word in words]
        filename = f"{'_'.join(safe_words).replace(' ', '_').lower()}_response.md"
    
    with open(filename, "w") as f:
        f.write(f"# Response to: {args.request[:50]}{'...' if len(args.request) > 50 else ''}\n\n")
        f.write(f"*Provided by: {response['agent']}*\n\n")
        f.write(response['content'])
    
    print(f"\nResponse saved to {filename}")


if __name__ == "__main__":
    main()