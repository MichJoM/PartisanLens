#!/usr/bin/env python3
"""
Main file to test the load_persona_and_headline_prompts function.
"""

import os
from utils import load_persona_and_headline_prompts


def test_load_persona_function():
    """Test the load_persona_and_headline_prompts function with sample data."""
    
    # Define test parameters
    persona_file_path = "/mnt/gpu-fastdata/anxo/PartisanLens/persona-llms/prompts/persona_simulation_prompts/ita/persona_1.md"
    test_headline = "Migranti, sbarchi senza fine: Lampedusa in tilt mentre la Germania ci rifila l'ennesima fregatura"
    
    print("="*60)
    print("Testing load_persona_and_headline_prompts function")
    print("="*60)
    
    # Check if persona file exists
    if not os.path.exists(persona_file_path):
        print(f"❌ Error: Persona file not found at {persona_file_path}")
        return
    
    print(f"📁 Persona file: {persona_file_path}")
    print(f"📰 Test headline: {test_headline}")
    print("\n" + "-"*60)
    
    try:
        # Call the function
        system_prompt, user_prompt = load_persona_and_headline_prompts(persona_file_path, test_headline)
        
        print("✅ Function executed successfully!")
        print("\n" + "="*60)
        print("SYSTEM PROMPT:")
        print("="*60)
        print(system_prompt)
        
        print("\n" + "="*60)
        print("USER PROMPT:")
        print("="*60)
        print(user_prompt)
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")


def test_multiple_personas():
    """Test the function with multiple persona files."""
    
    print("\n" + "="*60)
    print("Testing with multiple persona files")
    print("="*60)
    
    # Define test personas from different languages
    test_personas = [
        "/mnt/gpu-fastdata/anxo/PartisanLens/persona-llms/prompts/persona_simulation_prompts/ita/persona_1.md",
        "/mnt/gpu-fastdata/anxo/PartisanLens/persona-llms/prompts/persona_simulation_prompts/pt/persona_3.md",
        "/mnt/gpu-fastdata/anxo/PartisanLens/persona-llms/prompts/persona_simulation_prompts/spa/persona_2.md"
    ]
    
    test_headlines = [
        "Ripartizione dei migranti, cosa prevede Dublino",
        "Imigração: desafios e oportunidades para o mercado de trabalho", 
        "La UE debate nuevas políticas migratorias para 2024"
    ]
    
    for i, (persona_path, headline) in enumerate(zip(test_personas, test_headlines), 1):
        print(f"\n🔹 Test {i}:")
        print(f"  Persona: {os.path.basename(persona_path)}")
        print(f"  Language: {persona_path.split('/')[-2]}")
        print(f"  Headline: {headline}")
        
        if os.path.exists(persona_path):
            try:
                system_prompt, user_prompt = load_persona_and_headline_prompts(persona_path, headline)
                print("  ✅ Success!")
                print(f"  System prompt length: {len(system_prompt)} characters")
                print(f"  User prompt length: {len(user_prompt)} characters")
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
        else:
            print(f"  ❌ File not found: {persona_path}")


def show_function_info():
    """Display information about the function being tested."""
    
    print("="*60)
    print("FUNCTION INFORMATION")
    print("="*60)
    print("Function: load_persona_and_headline_prompts")
    print("Purpose: Load persona prompt from a file and combine it with a headline prompt")
    print("\nParameters:")
    print("  - persona_file_path (str): Path to the persona prompt file")
    print("  - headline (str): The headline to be analyzed")
    print("\nReturns:")
    print("  - system_prompt (str): Combined persona content with JSON format instructions")
    print("  - user_prompt (str): Combined persona and general guidelines with headline")
    print("="*60)


if __name__ == "__main__":
    # Show function information
    show_function_info()
    
    # Test the function with a single persona
    test_load_persona_function()
    
    # Test with multiple personas
    test_multiple_personas()
    
    print("\n" + "="*60)
    print("🎉 Testing completed!")
    print("="*60)
