#!/usr/bin/env python3
"""
Simple test file to quickly verify the load_persona_and_headline_prompts function.
"""

from utils import load_persona_and_headline_prompts


def simple_test():
    """Simple test of the function with clear output."""
    
    # Test parameters
    persona_file = "/mnt/gpu-fastdata/anxo/PartisanLens/persona-llms/prompts/persona_simulation_prompts/ita/persona_1.md"
    headline = "Ripartizione dei migranti, cosa prevede Dublino"
    
    print("🧪 Simple Test of load_persona_and_headline_prompts")
    print("=" * 50)
    print(f"Persona: {persona_file.split('/')[-1]}")
    print(f"Headline: {headline}")
    print("-" * 50)
    
    try:
        system_prompt, user_prompt = load_persona_and_headline_prompts(persona_file, headline)
        
        print("✅ SUCCESS!")
        print(f"📊 System prompt length: {len(system_prompt):,} characters")
        print(f"📊 User prompt length: {len(user_prompt):,} characters")
        
        # Show a preview of each prompt
        print("\n🔍 SYSTEM PROMPT PREVIEW:")
        print("-" * 50)
        print(system_prompt)
        
        print("\n🔍 USER PROMPT PREVIEW:")
        print("-" * 50)
        print(user_prompt)
        
        # Check if headline was properly inserted
        if headline in user_prompt:
            print(f"\n✅ Headline correctly inserted into user prompt")
        else:
            print(f"\n❌ WARNING: Headline not found in user prompt")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False
        
    return True


if __name__ == "__main__":
    simple_test()
