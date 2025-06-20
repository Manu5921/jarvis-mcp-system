/**
 * Système de validation automatique pour les réponses de Jarvis
 * Force l'utilisation des outils MCP et valide la qualité des analyses
 */

class JarvisResponseValidator {
    constructor() {
        this.forbiddenPhrases = [
            'probablement', 'semble', 'peut-être', 'il se peut que',
            'on peut supposer', 'il est possible que', 'vraisemblablement'
        ];
        
        this.requiredToolsUsage = ['LS', 'Read', 'Glob', 'Grep'];
        this.minimumScore = 7;
    }

    /**
     * Valide une réponse de Jarvis selon nos critères
     */
    async validateResponse(response, problemId) {
        const validation = {
            score: 0,
            errors: [],
            warnings: [],
            passed: false,
            feedback: []
        };

        // 1. Vérification des phrases interdites
        const forbiddenFound = this.checkForbiddenPhrases(response.content);
        if (forbiddenFound.length > 0) {
            validation.errors.push(`Phrases interdites détectées: ${forbiddenFound.join(', ')}`);
            validation.score -= 3;
        }

        // 2. Vérification utilisation des outils
        const toolsUsed = this.extractToolsUsed(response.toolsCalled || []);
        const missingTools = this.requiredToolsUsage.filter(tool => !toolsUsed.includes(tool));
        
        if (missingTools.length > 0) {
            validation.errors.push(`Outils manquants: ${missingTools.join(', ')}`);
            validation.score -= 4;
        } else {
            validation.score += 4;
            validation.feedback.push('✅ Utilisation correcte des outils MCP');
        }

        // 3. Vérification contenu factuel
        const factualContent = this.analyzeFactualContent(response.content);
        if (factualContent.factsCount < 3) {
            validation.warnings.push('Contenu insuffisamment factuel (< 3 faits concrets)');
            validation.score -= 1;
        } else {
            validation.score += 2;
        }

        // 4. Vérification structure de réponse
        const structureScore = this.validateResponseStructure(response.content);
        validation.score += structureScore;

        // 5. Score final et validation
        validation.score = Math.max(0, Math.min(10, validation.score + 5)); // Base de 5 points
        validation.passed = validation.score >= this.minimumScore;

        return validation;
    }

    /**
     * Détecte les phrases interdites dans la réponse
     */
    checkForbiddenPhrases(content) {
        return this.forbiddenPhrases.filter(phrase => 
            content.toLowerCase().includes(phrase.toLowerCase())
        );
    }

    /**
     * Extrait les outils utilisés depuis les appels de fonction
     */
    extractToolsUsed(toolsCalled) {
        const toolsMap = {
            'LS': ['ls', 'list'],
            'Read': ['read'],
            'Glob': ['glob'],
            'Grep': ['grep', 'search']
        };
        
        const usedTools = [];
        
        for (const [tool, aliases] of Object.entries(toolsMap)) {
            if (toolsCalled.some(call => aliases.includes(call.toLowerCase()))) {
                usedTools.push(tool);
            }
        }
        
        return usedTools;
    }

    /**
     * Analyse le contenu factuel de la réponse
     */
    analyzeFactualContent(content) {
        // Recherche d'indicateurs de contenu factuel
        const factIndicators = [
            /version\s+\d+\.\d+/gi,           // Versions
            /port\s+\d+/gi,                  // Ports
            /\w+\.json/gi,                   // Fichiers config
            /\/[\w\-\/]+\//gi,               // Chemins de fichiers
            /\b\w+\.js\b|\b\w+\.ts\b/gi,     // Fichiers code
            /dependencies|devDependencies/gi, // Dépendances
        ];
        
        let factsCount = 0;
        factIndicators.forEach(indicator => {
            const matches = content.match(indicator);
            if (matches) factsCount += matches.length;
        });
        
        return { factsCount };
    }

    /**
     * Valide la structure de la réponse
     */
    validateResponseStructure(content) {
        let score = 0;
        
        // Présence de sections structurées
        if (content.includes('##') || content.includes('**')) score += 1;
        
        // Listes ou énumérations
        if (content.includes('-') || content.includes('*') || content.includes('1.')) score += 1;
        
        // Longueur appropriée (ni trop courte, ni trop longue)
        if (content.length > 200 && content.length < 2000) score += 1;
        
        return score;
    }

    /**
     * Génère un feedback de correction
     */
    generateFeedback(validation, problemId) {
        let feedback = `\n🎯 **VALIDATION PROBLÈME ${problemId}**\n\n`;
        feedback += `📊 **Score: ${validation.score}/10** ${validation.passed ? '✅ VALIDÉ' : '❌ ÉCHEC'}\n\n`;
        
        if (validation.errors.length > 0) {
            feedback += `❌ **Erreurs critiques:**\n`;
            validation.errors.forEach(error => feedback += `- ${error}\n`);
            feedback += `\n`;
        }
        
        if (validation.warnings.length > 0) {
            feedback += `⚠️ **Avertissements:**\n`;
            validation.warnings.forEach(warning => feedback += `- ${warning}\n`);
            feedback += `\n`;
        }
        
        if (validation.feedback.length > 0) {
            feedback += `✅ **Points positifs:**\n`;
            validation.feedback.forEach(point => feedback += `- ${point}\n`);
            feedback += `\n`;
        }
        
        if (!validation.passed) {
            feedback += `🔧 **Actions requises:**\n`;
            feedback += `1. Utiliser OBLIGATOIREMENT LS/Read/Glob pour explorer\n`;
            feedback += `2. Éliminer toutes les suppositions\n`;
            feedback += `3. Baser la réponse uniquement sur les faits découverts\n`;
            feedback += `4. Structurer la réponse avec sections claires\n\n`;
            feedback += `🔄 **Reprendre l'analyse avec la méthode correcte**\n`;
        }
        
        return feedback;
    }
}

// Export pour utilisation
module.exports = { JarvisResponseValidator };

/**
 * Exemple d'utilisation:
 * 
 * const validator = new JarvisResponseValidator();
 * const validation = await validator.validateResponse(jarvisResponse, 'problem_01');
 * const feedback = validator.generateFeedback(validation, 'problem_01');
 * 
 * if (!validation.passed) {
 *     // Envoyer feedback de correction à Jarvis
 *     await sendCorrectionToJarvis(feedback);
 * }
 */