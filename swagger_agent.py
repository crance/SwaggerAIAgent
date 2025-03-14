import json
import yaml
import ollama
import argparse
import os
import re
import requests
from rich.console import Console
from rich.markdown import Markdown
from openapi_spec_validator import validate
from openapi_spec_validator.exceptions import OpenAPISpecValidatorError

console = Console()

DEFAULT_MODEL = "deepseek-coder:6.7b"
class SwaggerAPIAgent:
    def __init__(self, model_name=None, modelfile_path=None, ollama_host="http://localhost:11434", swagger_file=None):
        self.ollama_host = ollama_host
        self.swagger_data = None
        self.endpoints = []
        self.endpoint_relationships = {}
        
        # Handle model and modelfile parameters
        if modelfile_path:
            # If modelfile is provided, create model from it and use that
            model_name = self._create_model_from_file(modelfile_path, model_name)
            self.model = model_name
        elif model_name:
            # If only model name is provided, use that
            self.model = model_name
        else:
            # No model specified, use a default
            self.model = DEFAULT_MODEL
            console.print("[yellow]No model or modelfile specified, using default model '{DEFAULT_MODEL}'[/yellow]")

        if swagger_file:
            self.load_swagger(swagger_file)
    
    def _create_model_from_file(self, modelfile_path, model_name=None):
        """Create a model from modelfile and return the model name"""
        try:
            # Read the modelfile content
            with open(modelfile_path, 'r') as file:
                modelfile_content = file.read()
            
            # Extract model name from modelfile if not provided
            if not model_name:
                # Try to extract model name from the FROM line in the modelfile
                match = re.search(r'^FROM\s+([^\s]+)', modelfile_content, re.MULTILINE)
                if match:
                    base_model = match.group(1)
                    model_name = f"swagger-{base_model}-agent"

                    # Extract system prompt
                    system_pattern = r'SYSTEM\s+"""([\s\S]*?)"""\s*'
                    system_match = re.search(system_pattern, modelfile_content, re.DOTALL)
                    system_prompt = system_match.group(1).strip() if system_match else None

                    # Extract template
                    template_pattern = r'TEMPLATE\s+"""([\s\S]*?)"""\s*'
                    template_match = re.search(template_pattern, modelfile_content, re.DOTALL)
                    template = template_match.group(1) if template_match else None

                    # Extract parameters
                    parameter_pattern = r'PARAMETER\s+(\w+)\s+([^\n\r]+)'
                    parameter_matches = re.finditer(parameter_pattern, modelfile_content)

                    parameters = {}
                    stop_tokens = []
                    for match in parameter_matches:
                        param_name = match.group(1)
                        param_value = match.group(2).strip()
                        
                        # Handle quoted values
                        if (param_value.startswith('"') and param_value.endswith('"')) or \
                        (param_value.startswith("'") and param_value.endswith("'")):
                            param_value = param_value[1:-1]  # Remove quotes
                        
                        # Special handling for stop tokens
                        if param_name == "stop":
                            stop_tokens.append(param_value)
                        else:
                            # Convert numeric values to appropriate types
                            try:
                                if '.' in param_value:
                                    param_value = float(param_value)
                                else:
                                    param_value = int(param_value)
                            except ValueError:
                                # Keep as string if not a number
                                pass
                                
                            parameters[param_name] = param_value
                    
                    # Add the stop tokens to parameters if any were found
                    if stop_tokens:
                        parameters["stop"] = stop_tokens
                        
                else:
                    # Use a default name based on the filename
                    base_name = os.path.basename(modelfile_path).split('.')[0]
                    model_name = f"swagger-{base_name}-agent"

            # Prepare the request data
            data = {
                "model": model_name,
                "from": base_model,
                "system": system_prompt,
                "template": template,
                "parameters": parameters
            }

            # console.print(f"[blue]Sending request data: {json.dumps(data, indent=2)}[/blue]")
            
            # Make the API request to create/update the model
            response = requests.post(
                f"{self.ollama_host}/api/create",
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                console.print(f"[green]Successfully created/updated model '{model_name}' from modelfile[/green]")
                console.print(f"[green]'{response.text}'[/green]")
            else:
                console.print(f"[red]Failed to create/update model: {response.text}[/red]")
                console.print(f"[yellow]Falling back to default model '{DEFAULT_MODEL}'[/yellow]")
                return DEFAULT_MODEL
                
            return model_name
        except Exception as e:
            console.print(f"[red]Error creating model from modelfile: {str(e)}[/red]")
            console.print(f"[yellow]Falling back to default model '{DEFAULT_MODEL}'[/yellow]")
            return DEFAULT_MODEL

    def load_swagger(self, swagger_file):
        """Load and parse a Swagger/OpenAPI specification file"""
        try:
            if swagger_file.startswith("http://") or swagger_file.startswith("https://"):
                response = requests.get(swagger_file)
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'json' in content_type:
                        self.swagger_data = response.json()
                    elif 'yaml' in content_type or 'yml' in content_type:
                        self.swagger_data = yaml.safe_load(response.text)
                    else:
                        # Try to parse as JSON first, then YAML
                        try:
                            self.swagger_data = response.json()
                        except:
                            self.swagger_data = yaml.safe_load(response.text)
                else:
                    console.print(f"Failed to fetch Swagger file from URL: {swagger_file}", style="bold red")
                    return
            elif swagger_file.endswith(".json"):
                with open(swagger_file, "r") as file:
                    self.swagger_data = json.load(file)
            elif swagger_file.endswith(".yaml") or swagger_file.endswith(".yml"):
                with open(swagger_file, "r") as file:
                    self.swagger_data = yaml.safe_load(file)
            else:
                console.print(f"Unsupported file format: {swagger_file}", style="bold red")
                return

            # Validate the loaded Swagger data
            try:
                # The validate function handles both OpenAPI 2.0 and 3.0
                validate(self.swagger_data)
                console.print("API specification validated successfully.", style="bold green")
            except OpenAPISpecValidatorError as e:
                console.print(f"Warning: Specification validation error: {e}", style="bold yellow")
                console.print("Continuing with invalid specification...", style="yellow")

            self._extract_endpoints()
            self._analyze_endpoint_relationships()
            
            console.print(f"[green]Successfully loaded API specification from {swagger_file}[/green]")
            console.print(f"[blue]Found {len(self.endpoints)} endpoints[/blue]")

        except Exception as e:
            console.print(f"Error loading Swagger specification file: {e}", style="bold red")
            self.swagger_data = None

    def _extract_endpoints(self):
        """Extract endpoints from the swagger specification"""
        self.endpoints = []
            
        if not self.swagger_data:
            return
            
        # Handle both OpenAPI 2.0 (Swagger) and OpenAPI 3.0
        if 'paths' in self.swagger_data:
            for path, methods in self.swagger_data['paths'].items():
                for method, details in methods.items():
                    if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                        endpoint = {
                            'path': path,
                            'method': method.upper(),
                            'summary': details.get('summary', ''),
                            'description': details.get('description', ''),
                            'parameters': details.get('parameters', []),
                            'responses': details.get('responses', {})
                        }
                            
                        # Handle request body for OpenAPI 3.0
                        if 'requestBody' in details:
                            endpoint['requestBody'] = details['requestBody']
                        
                        # Extract schema information for request and response
                        schemas = self._extract_schemas(details)
                        if schemas:
                            endpoint['schemas'] = schemas
                                
                        self.endpoints.append(endpoint)

    def _extract_schemas(self, details):
        """Extract schema information from endpoint details"""
        schemas = {'request': {}, 'response': {}}
        
        # Extract request schema
        if 'requestBody' in details and 'content' in details['requestBody']:
            for content_type, content_details in details['requestBody']['content'].items():
                if 'schema' in content_details:
                    schemas['request'][content_type] = content_details['schema']
        
        # Extract response schemas
        if 'responses' in details:
            for status, response_details in details['responses'].items():
                if 'content' in response_details:
                    for content_type, content_details in response_details['content'].items():
                        if 'schema' in content_details:
                            if status not in schemas['response']:
                                schemas['response'][status] = {}
                            schemas['response'][status][content_type] = content_details['schema']
        
        return schemas if (schemas['request'] or schemas['response']) else None

    def _analyze_endpoint_relationships(self):
        """Analyze relationships between endpoints to identify potential multi-step workflows"""
        if not self.endpoints:
            return
        
        # Reset relationships
        self.endpoint_relationships = {}
        
        # Identify potential data flow between endpoints
        for i, endpoint1 in enumerate(self.endpoints):
            self.endpoint_relationships[i] = []
            
            # Get response schemas for this endpoint
            response_schemas = self._get_response_schemas(endpoint1)
            
            for j, endpoint2 in enumerate(self.endpoints):
                if i == j:
                    continue
                
                # Get required request parameters/body for the second endpoint
                request_params = self._get_request_params(endpoint2)
                
                # Check if there's potential data flow from endpoint1 to endpoint2
                if self._check_data_compatibility(response_schemas, request_params):
                    self.endpoint_relationships[i].append(j)
    
    def _get_response_schemas(self, endpoint):
        """Get response schema properties from an endpoint"""
        properties = {}
        
        # Extract from OpenAPI 3.0 structure
        if 'schemas' in endpoint and 'response' in endpoint['schemas']:
            for status, content_types in endpoint['schemas']['response'].items():
                if status.startswith('2'):  # Consider successful responses (2xx)
                    for content_type, schema in content_types.items():
                        props = self._extract_schema_properties(schema)
                        if props:
                            properties.update(props)
        
        # Extract from responses for both OpenAPI 2.0 and 3.0
        if 'responses' in endpoint:
            for status, response in endpoint['responses'].items():
                if status.startswith('2'):  # Consider successful responses
                    if 'schema' in response:
                        props = self._extract_schema_properties(response['schema'])
                        if props:
                            properties.update(props)
        
        return properties
    
    def _get_request_params(self, endpoint):
        """Get required request parameters and body properties for an endpoint"""
        params = {}
        
        # Extract from parameters
        if 'parameters' in endpoint:
            for param in endpoint['parameters']:
                if param.get('required', False):
                    params[param.get('name')] = param.get('type', 'string')
        
        # Extract from requestBody for OpenAPI 3.0
        if 'schemas' in endpoint and 'request' in endpoint['schemas']:
            for content_type, schema in endpoint['schemas']['request'].items():
                props = self._extract_schema_properties(schema)
                if props:
                    params.update(props)
        
        # Extract from parameters with in: "body" for OpenAPI 2.0
        if 'parameters' in endpoint:
            for param in endpoint['parameters']:
                if param.get('in') == 'body' and 'schema' in param:
                    props = self._extract_schema_properties(param['schema'])
                    if props:
                        params.update(props)
        
        return params
    
    def _extract_schema_properties(self, schema):
        """Extract properties from a schema object"""
        properties = {}
        
        if not schema:
            return properties
            
        # Handle $ref
        if '$ref' in schema:
            ref_schema = self._resolve_reference(schema['$ref'])
            if ref_schema:
                return self._extract_schema_properties(ref_schema)
        
        # Extract from properties
        if 'properties' in schema:
            for prop_name, prop_details in schema['properties'].items():
                properties[prop_name] = prop_details.get('type', 'object')
        
        # Handle arrays
        if schema.get('type') == 'array' and 'items' in schema:
            if '$ref' in schema['items']:
                item_schema = self._resolve_reference(schema['items']['$ref'])
                if item_schema and 'properties' in item_schema:
                    for prop_name, prop_details in item_schema['properties'].items():
                        properties[f"items.{prop_name}"] = prop_details.get('type', 'object')
            elif 'properties' in schema['items']:
                for prop_name, prop_details in schema['items']['properties'].items():
                    properties[f"items.{prop_name}"] = prop_details.get('type', 'object')
        
        return properties
    
    def _resolve_reference(self, ref):
        """Resolve a JSON reference in the Swagger/OpenAPI spec"""
        if not ref.startswith('#/'):
            return None  # External references not supported
        
        path = ref[2:].split('/')
        current = self.swagger_data
        
        for component in path:
            if component in current:
                current = current[component]
            else:
                return None
        
        return current
    
    def _check_data_compatibility(self, response_props, request_params):
        """Check if response properties from one endpoint can satisfy request parameters of another"""
        if not response_props or not request_params:
            return False
        
        # Simple compatibility check: are there any matching property names?
        for param_name in request_params.keys():
            if param_name in response_props:
                return True
        
        return False

    def get_endpoints_summary(self):
        """Get a summary of all endpoints for the agent's context"""
        if not self.endpoints:
            return "No API endpoints loaded."
            
        summary = "API Endpoints Summary:\n\n"
        for i, endpoint in enumerate(self.endpoints):
            summary += f"{i+1}. {endpoint['method']} {endpoint['path']}"
            if endpoint['summary']:
                summary += f" - {endpoint['summary']}"
            summary += "\n"
            
        return summary

    def get_detailed_endpoint_info(self, index):
        """Get detailed information about a specific endpoint"""
        if index < 0 or index >= len(self.endpoints):
            return "Invalid endpoint index."
            
        endpoint = self.endpoints[index]
        
        details = f"## {endpoint['method']} {endpoint['path']}\n\n"
        
        if endpoint['summary']:
            details += f"**Summary:** {endpoint['summary']}\n\n"
            
        if endpoint['description']:
            details += f"**Description:** {endpoint['description']}\n\n"
            
        # Parameters
        if endpoint['parameters']:
            details += "**Parameters:**\n\n"
            for param in endpoint['parameters']:
                required = " (Required)" if param.get('required', False) else " (Optional)"
                details += f"- `{param.get('name')}` ({param.get('in', 'unknown')}): {param.get('description', 'No description')}{required}\n"
            details += "\n"
            
        # Request Body for OpenAPI 3.0
        if 'requestBody' in endpoint:
            details += "**Request Body:**\n\n"
            if 'description' in endpoint['requestBody']:
                details += f"{endpoint['requestBody']['description']}\n\n"
            if 'content' in endpoint['requestBody']:
                for content_type, content_info in endpoint['requestBody']['content'].items():
                    details += f"Content Type: `{content_type}`\n\n"
                    if 'schema' in content_info:
                        schema_info = self._get_schema_description(content_info['schema'])
                        if schema_info:
                            details += f"{schema_info}\n\n"
            
        # Responses
        if endpoint['responses']:
            details += "**Responses:**\n\n"
            for status, response in endpoint['responses'].items():
                details += f"- `{status}`: {response.get('description', 'No description')}\n"
                
                # For OpenAPI 3.0
                if 'content' in response:
                    for content_type, content_info in response['content'].items():
                        details += f"  Content Type: `{content_type}`\n"
                        if 'schema' in content_info:
                            schema_info = self._get_schema_description(content_info['schema'])
                            if schema_info:
                                details += f"  Schema: {schema_info}\n"
                                
                # For OpenAPI 2.0
                if 'schema' in response:
                    schema_info = self._get_schema_description(response['schema'])
                    if schema_info:
                        details += f"  Schema: {schema_info}\n"
            
        # Related endpoints (potential workflow steps)
        if index in self.endpoint_relationships and self.endpoint_relationships[index]:
            details += "\n**Potential Next Steps:**\n\n"
            for related_index in self.endpoint_relationships[index]:
                related = self.endpoints[related_index]
                details += f"- {related_index+1}. {related['method']} {related['path']} - {related['summary']}\n"
        
        return details
    
    def _get_schema_description(self, schema):
        """Get a human-readable description of a schema"""
        if not schema:
            return None
            
        # Handle $ref
        if '$ref' in schema:
            ref_schema = self._resolve_reference(schema['$ref'])
            if ref_schema:
                ref_name = schema['$ref'].split('/')[-1]
                return f"`{ref_name}` object"
        
        # Handle array
        if schema.get('type') == 'array' and 'items' in schema:
            items_desc = self._get_schema_description(schema['items'])
            return f"Array of {items_desc}" if items_desc else "Array of items"
        
        # Handle object
        if schema.get('type') == 'object' or 'properties' in schema:
            if 'properties' in schema and schema['properties']:
                return "Object with properties"
            return "Object"
        
        # Handle primitive types
        if 'type' in schema:
            return f"{schema['type']}"
        
        return "Unknown schema"

    def query(self, user_query):
        """Process a user query through the Ollama model"""
        if not self.swagger_data:
            return "Please load a Swagger/OpenAPI specification first."
        
        # Create context with endpoints, their relationships, and detailed information
        endpoints_summary = self.get_endpoints_summary()
        
        # Add more detailed information about potentially relevant endpoints
        potential_workflows = self._get_potential_workflow_sequences()
        
        # Build a comprehensive context for the AI
        context = f"""
API Endpoints Summary:
{endpoints_summary}

Detailed information for key endpoints:
{self._get_detailed_info_for_key_endpoints(5)}

Potential API Workflow Sequences:
{potential_workflows}

API Base URL: {self._get_api_base_url()}

Authentication: {self._get_authentication_info_short()}
"""
        
        prompt = f"""
Here's the summary of available API endpoints and their relationships:

{context}

User question: {user_query}

Please analyze this API to determine which endpoint(s) would best serve the user's needs. If a single API call is sufficient, recommend that approach. If a multi-step workflow is needed, explain the sequence and data flow between calls. Always include complete, production-ready sample code showing how to implement your recommendation, with proper error handling and security practices.
"""

        try:
            client = ollama.Client(host=self.ollama_host)
            response = client.chat(model=self.model, messages=[
                {"role": "user", "content": prompt}
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error querying the model: {str(e)}"
    
    def _get_api_base_url(self):
        """Extract the base URL from the Swagger/OpenAPI spec"""
        if 'servers' in self.swagger_data and self.swagger_data['servers']:
            # OpenAPI 3.0
            return self.swagger_data['servers'][0].get('url', 'https://api.example.com')
        elif 'host' in self.swagger_data:
            # Swagger 2.0
            scheme = self.swagger_data.get('schemes', ['https'])[0]
            host = self.swagger_data.get('host', 'api.example.com')
            basePath = self.swagger_data.get('basePath', '/')
            return f"{scheme}://{host}{basePath}"
        else:
            return "https://api.example.com"
    
    def _get_authentication_info_short(self):
        """Get a short summary of authentication methods"""
        # For OpenAPI 3.0
        if 'components' in self.swagger_data and 'securitySchemes' in self.swagger_data['components']:
            schemes = []
            for name, scheme in self.swagger_data['components']['securitySchemes'].items():
                schemes.append(f"{name} ({scheme.get('type', 'unknown')})")
            return ", ".join(schemes) if schemes else "None specified"
        
        # For OpenAPI 2.0 (Swagger)
        if 'securityDefinitions' in self.swagger_data:
            schemes = []
            for name, scheme in self.swagger_data['securityDefinitions'].items():
                schemes.append(f"{name} ({scheme.get('type', 'unknown')})")
            return ", ".join(schemes) if schemes else "None specified"
        
        return "None specified"
    
    def _get_detailed_info_for_key_endpoints(self, max_endpoints=5):
        """Get detailed information for a subset of important endpoints"""
        if not self.endpoints:
            return "No endpoints available."
            
        # For now, just select the first few endpoints
        # This could be improved with better heuristics for endpoint importance
        selected_indices = list(range(min(max_endpoints, len(self.endpoints))))
        
        details = ""
        for idx in selected_indices:
            endpoint = self.endpoints[idx]
            details += f"Endpoint {idx+1}: {endpoint['method']} {endpoint['path']}\n"
            details += f"Summary: {endpoint['summary']}\n"
            if idx in self.endpoint_relationships and self.endpoint_relationships[idx]:
                details += "Related endpoints: " + ", ".join([str(i+1) for i in self.endpoint_relationships[idx]]) + "\n"
            details += "\n"
            
        return details
    
    def _get_potential_workflow_sequences(self):
        """Identify potential workflow sequences based on endpoint relationships"""
        if not self.endpoint_relationships:
            return "No endpoint relationships identified."
            
        workflows = []
        
        # Find chains of at least 2 endpoints
        for start_idx in self.endpoint_relationships:
            if self.endpoint_relationships[start_idx]:
                for next_idx in self.endpoint_relationships[start_idx]:
                    workflow = [
                        f"{start_idx+1}. {self.endpoints[start_idx]['method']} {self.endpoints[start_idx]['path']}",
                        f"{next_idx+1}. {self.endpoints[next_idx]['method']} {self.endpoints[next_idx]['path']}"
                    ]
                    
                    # See if we can extend the chain further
                    if next_idx in self.endpoint_relationships and self.endpoint_relationships[next_idx]:
                        third_idx = self.endpoint_relationships[next_idx][0]  # Just take the first one
                        workflow.append(f"{third_idx+1}. {self.endpoints[third_idx]['method']} {self.endpoints[third_idx]['path']}")
                    
                    workflows.append(" → ".join(workflow))
                    
                    # Limit to avoid overwhelming the context
                    if len(workflows) >= 5:
                        break
            
            if len(workflows) >= 5:
                break
        
        if not workflows:
            return "No multi-step workflows identified."
            
        return "\n".join(workflows)


def main():
    parser = argparse.ArgumentParser(description="Swagger API Agent")
    parser.add_argument("--swagger", type=str,default="https://petstore.swagger.io/v2/swagger.json", help="Path to Swagger/OpenAPI specification file or URL")
    parser.add_argument("--model", type=str, help="Ollama model name")
    parser.add_argument("--modelfile", type=str, default= "models/Modelfile.swaggeragent", help="Path to Ollama modelfile (if provided, takes precedence over model)")
    # parser.add_argument("--modelfile", type=str, help="Path to Ollama modelfile (if provided, takes precedence over model)")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="Ollama API host URL")
    args = parser.parse_args()

    agent = SwaggerAPIAgent(
        model_name=args.model,
        modelfile_path=args.modelfile,
        ollama_host=args.ollama_host
    )

    if args.swagger:
        agent.load_swagger(args.swagger)

    console.print("[bold green]Swagger API Agent[/bold green]")
    console.print("[blue]Ask any question about using this API, and I'll help you find the right endpoints and generate code.[/blue]")
    console.print("[bold]Examples:[/bold]")
    console.print("  • How do I create a new user account?")
    console.print("  • What's the best way to search for products?")
    console.print("  • [bold blue]exit[/bold blue] or [bold blue]quit[/bold blue]: End the session")
    console.print("  • [bold blue]load <file_path_or_url>[/bold blue]: Load a different API specification")
    
    while True:
        user_input = input("\n> ")

        if user_input.lower() in ['exit', 'quit']:
            break
        elif user_input.lower().startswith('load '):
            file_path = user_input[5:].strip()
            agent.load_swagger(file_path)
        else:
            # All other inputs are treated as natural language queries about the API
            response = agent.query(user_input)
            console.print(Markdown(response))


if __name__ == "__main__":
    main()