/**
 * Orchestrateur de formation pour Jarvis MCP
 * Lance les 10 problèmes progressifs avec validation automatique
 */

const { JarvisResponseValidator } = require('./auto-validator.js');

class JarvisTrainingOrchestrator {
    constructor() {
        this.validator = new JarvisResponseValidator();
        this.currentProblem = 1;
        this.maxProblems = 10;
        this.maxRetries = 2;
        this.mcpHubUrl = 'http://localhost:4000';
        this.trainingLog = [];
    }

    /**
     * Lance le programme de formation complet
     */
    async startTraining() {
        console.log('🎓 DÉMARRAGE FORMATION JARVIS MCP');
        console.log('=====================================');
        
        // 1. Configuration des prompts système
        await this.configureSystemPrompts();
        
        // 2. Envoi des règles de formation
        await this.sendTrainingRules();
        
        // 3. Lancement des problèmes progressifs
        for (let problemId = 1; problemId <= this.maxProblems; problemId++) {
            console.log(`\n🎯 PROBLÈME ${problemId}/${this.maxProblems}`);
            
            const success = await this.runProblem(problemId);
            
            if (!success) {
                console.log(`❌ Échec définitif sur le problème ${problemId}`);
                break;
            }
            
            console.log(`✅ Problème ${problemId} validé !`);
        }
        
        // 4. Rapport final
        await this.generateFinalReport();
    }

    /**
     * Configure les prompts système améliorés
     */
    async configureSystemPrompts() {
        console.log('🔧 Configuration des prompts système...');
        
        const systemConfig = {
            message: `CONFIGURATION FORMATION OBLIGATOIRE:
            
            RÈGLES ABSOLUES:
            1. TOUJOURS explorer avec LS/Read/Glob avant toute analyse
            2. INTERDICTION totale des suppositions (probablement, semble, peut-être)
            3. WORKFLOW OBLIGATOIRE: Explorer → Analyser → Synthétiser
            4. Réponses basées UNIQUEMENT sur faits découverts
            
            VALIDATION AUTOMATIQUE ACTIVE:
            - Score minimum requis: 7/10
            - Maximum 2 tentatives par problème
            - Feedback correctif automatique en cas d'échec
            
            FORMATION EN COURS - APPRENTISSAGE OBLIGATOIRE`,
            session_id: 'training-system-config',
            context: 'Configuration formation'
        };
        
        await this.sendToMCP(systemConfig);
    }

    /**
     * Envoie les règles de formation à Jarvis
     */
    async sendTrainingRules() {
        console.log('📋 Envoi des règles de formation...');
        
        const rules = {
            message: `PROGRAMME FORMATION - 10 PROBLÈMES PROGRESSIFS:

            NIVEAUX:
            - Problèmes 1-2: Débutant (structure, technologies)
            - Problèmes 3-5: Intermédiaire (architecture, workflows)  
            - Problèmes 6-8: Avancé (intégrations, tests)
            - Problèmes 9-10: Expert (projets complets, recommandations)
            
            VALIDATION:
            ✅ Chaque problème doit obtenir 7/10 minimum
            ✅ Utilisation obligatoire des outils MCP
            ✅ Contenu factuel uniquement
            ✅ Structure claire et précise
            
            PRÊT POUR FORMATION ?`,
            session_id: 'training-rules',
            context: 'Règles formation'
        };
        
        await this.sendToMCP(rules);
    }

    /**
     * Exécute un problème de formation
     */
    async runProblem(problemId) {
        const problems = await this.loadTrainingProblems();
        const problem = problems.training_problems.find(p => 
            p.id === `problem_${problemId.toString().padStart(2, '0')}`
        );
        
        if (!problem) {
            console.log(`❌ Problème ${problemId} non trouvé`);
            return false;
        }
        
        console.log(`📝 ${problem.title} (${problem.level})`);
        console.log(`🎯 ${problem.instruction}`);
        
        let attempt = 1;
        let success = false;
        
        while (attempt <= this.maxRetries && !success) {
            console.log(`\n🔄 Tentative ${attempt}/${this.maxRetries}`);
            
            // Envoi du problème à Jarvis
            const response = await this.sendProblemToJarvis(problem);
            
            // Validation automatique
            const validation = await this.validator.validateResponse(response, problem.id);
            
            if (validation.passed) {
                success = true;
                console.log(`✅ Score: ${validation.score}/10 - VALIDÉ !`);
                this.logTrainingResult(problemId, attempt, validation, true);
            } else {
                console.log(`❌ Score: ${validation.score}/10 - ÉCHEC`);
                
                // Génération et envoi du feedback correctif
                const feedback = this.validator.generateFeedback(validation, problem.id);
                await this.sendCorrectionFeedback(feedback, problem);
                
                this.logTrainingResult(problemId, attempt, validation, false);
                attempt++;
            }
        }
        
        return success;
    }

    /**
     * Envoie un problème à Jarvis
     */
    async sendProblemToJarvis(problem) {
        const problemMessage = {
            message: `PROBLÈME FORMATION ${problem.id.toUpperCase()}:

            ${problem.instruction}
            
            RAPPEL WORKFLOW OBLIGATOIRE:
            1. LS pour explorer la structure
            2. Read des fichiers clés
            3. Glob pour identifier patterns
            4. Analyse factuelle uniquement
            5. Synthèse structurée
            
            ATTENTION: Validation automatique active !`,
            session_id: `training-${problem.id}`,
            context: `Formation ${problem.level}`
        };
        
        console.log('📤 Envoi du problème à Jarvis...');
        const response = await this.sendToMCP(problemMessage);
        
        // Simulation de l'extraction des outils utilisés
        // Dans la vraie implémentation, on capturerait les appels MCP
        response.toolsCalled = this.extractToolsFromResponse(response.response);
        
        return response;
    }

    /**
     * Envoie un feedback correctif
     */
    async sendCorrectionFeedback(feedback, problem) {
        const correctionMessage = {
            message: `FEEDBACK CORRECTIF:

            ${feedback}
            
            REPRENDRE IMMÉDIATEMENT avec la méthode correcte:
            ${problem.instruction}`,
            session_id: `correction-${problem.id}`,
            context: 'Correction formation'
        };
        
        console.log('📋 Envoi du feedback correctif...');
        await this.sendToMCP(correctionMessage);
    }

    /**
     * Génère le rapport final de formation
     */
    async generateFinalReport() {
        console.log('\n📊 GÉNÉRATION RAPPORT FINAL');
        console.log('============================');
        
        const successRate = this.trainingLog.filter(log => log.success).length / this.trainingLog.length * 100;
        const averageScore = this.trainingLog.reduce((sum, log) => sum + log.score, 0) / this.trainingLog.length;
        
        const report = `RAPPORT FORMATION JARVIS MCP:

        📊 STATISTIQUES:
        - Problèmes complétés: ${this.trainingLog.length}/${this.maxProblems}
        - Taux de réussite: ${successRate.toFixed(1)}%
        - Score moyen: ${averageScore.toFixed(1)}/10
        
        🎯 PROGRESSION:
        ${this.trainingLog.map(log => 
            `Problème ${log.problemId}: ${log.score}/10 (${log.success ? '✅' : '❌'})`
        ).join('\n        ')}
        
        📈 AMÉLIORATIONS OBSERVÉES:
        ${this.analyzeImprovements()}
        
        FORMATION ${successRate >= 70 ? 'RÉUSSIE' : 'À REPRENDRE'} !`;
        
        console.log(report);
        
        // Envoi du rapport à Jarvis
        await this.sendToMCP({
            message: report,
            session_id: 'training-final-report',
            context: 'Rapport final formation'
        });
    }

    /**
     * Fonctions utilitaires
     */
    async loadTrainingProblems() {
        const fs = require('fs').promises;
        const content = await fs.readFile('./training-problems.json', 'utf8');
        return JSON.parse(content);
    }

    async sendToMCP(message) {
        // Simulation d'appel HTTP au hub MCP
        console.log(`📡 → MCP Hub: ${message.message.substring(0, 50)}...`);
        
        // Dans la vraie implémentation:
        // const response = await fetch(`${this.mcpHubUrl}/mcp/chat`, {
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify(message)
        // });
        // return await response.json();
        
        // Simulation de réponse
        return {
            response: `Réponse simulée pour: ${message.message.substring(0, 30)}...`,
            timestamp: new Date().toISOString(),
            session_id: message.session_id
        };
    }

    extractToolsFromResponse(responseText) {
        // Extraction des outils depuis le texte de réponse
        const tools = [];
        if (responseText.includes('LS') || responseText.includes('ls')) tools.push('LS');
        if (responseText.includes('Read') || responseText.includes('read')) tools.push('Read');
        if (responseText.includes('Glob') || responseText.includes('glob')) tools.push('Glob');
        if (responseText.includes('Grep') || responseText.includes('grep')) tools.push('Grep');
        return tools;
    }

    logTrainingResult(problemId, attempt, validation, success) {
        this.trainingLog.push({
            problemId,
            attempt,
            score: validation.score,
            success,
            errors: validation.errors,
            timestamp: new Date().toISOString()
        });
    }

    analyzeImprovements() {
        if (this.trainingLog.length < 2) return "Données insuffisantes";
        
        const firstScore = this.trainingLog[0].score;
        const lastScore = this.trainingLog[this.trainingLog.length - 1].score;
        const improvement = lastScore - firstScore;
        
        return improvement > 0 ? 
            `Amélioration de ${improvement.toFixed(1)} points` :
            `Régression de ${Math.abs(improvement).toFixed(1)} points`;
    }
}

// Point d'entrée
if (require.main === module) {
    const trainer = new JarvisTrainingOrchestrator();
    trainer.startTraining().catch(console.error);
}

module.exports = { JarvisTrainingOrchestrator };