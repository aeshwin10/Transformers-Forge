import matplotlib.pyplot as plt

# BLEU score values
models = ['My Model (71%)', 'ChatGPT 3.5 (88%)']
bleu_scores = [71, 88]
colors = ['blue', 'orange']

# Create bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(models, bleu_scores, color=colors)

# Labels and title
plt.xlabel('Model')
plt.ylabel('BLEU Score (%)')
plt.title('BLEU Score Comparison')
plt.ylim(0, 100)

# Annotate each bar with its score
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height}%', ha='center', va='bottom')

plt.tight_layout()
# Save and display
plt.savefig('bleu_score_comparison.png')
plt.show()
