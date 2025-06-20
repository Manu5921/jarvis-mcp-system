/**
 * Lanceur de formation pour Jarvis MCP
 * Évite les validations répétées de Claude Code
 */

import http from 'http';

class TrainingLauncher {
    constructor() {
        this.mcpHubUrl = 'http://localhost:4000';
        this.problems = [
            {
                id: 1,
                message: "FORMATION PROBLÈME 1: Analyse restaurant-app structure et framework. WORKFLOW: LS puis Read package.json puis analyse factuelle.",
                session: "training-01"
            },
            {
                id: 2,
                message: "FORMATION PROBLÈME 2: Quelles technologies dans agents/01-design-agent ? WORKFLOW: LS + Read configs + Glob patterns.",
                session: "training-02"
            },
            {
                id: 3,
                message: "FORMATION PROBLÈME 3: Comment fonctionne orchestrateur ? WORKFLOW: LS + Read master-config.json + analyse coordination.",
                session: "training-03"
            }
        ];
    }

    async sendProblem(problem) {
        const data = JSON.stringify({
            message: problem.message,
            session_id: problem.session,
            context: "Formation automatique"
        });

        const options = {
            hostname: 'localhost',
            port: 4000,
            path: '/mcp/chat',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': data.length
            }
        };

        return new Promise((resolve, reject) => {
            const req = http.request(options, (res) => {
                let responseData = '';
                
                res.on('data', (chunk) => {
                    responseData += chunk;
                });
                
                res.on('end', () => {
                    try {
                        const response = JSON.parse(responseData);
                        resolve(response);
                    } catch (e) {
                        resolve({ response: responseData });
                    }
                });
            });

            req.on('error', (error) => {
                reject(error);
            });

            req.write(data);
            req.end();
        });
    }

    async launchTraining() {
        console.log('🎓 FORMATION JARVIS MCP - LANCEMENT AUTOMATIQUE');
        console.log('===============================================');

        for (const problem of this.problems) {
            console.log(`\n🎯 PROBLÈME ${problem.id}`);
            console.log(`📝 ${problem.message}`);
            
            try {
                const response = await this.sendProblem(problem);
                console.log('✅ Envoyé avec succès');
                console.log(`📋 Réponse: ${response.response?.substring(0, 100)}...`);
                
                // Pause entre les problèmes
                await new Promise(resolve => setTimeout(resolve, 2000));
                
            } catch (error) {
                console.log(`❌ Erreur: ${error.message}`);
            }
        }

        console.log('\n🎯 FORMATION LANCÉE');
        console.log('📊 Surveillez les réponses de Jarvis pour validation');
    }
}

// Lancement si script exécuté directement
const launcher = new TrainingLauncher();
launcher.launchTraining().catch(console.error);

export { TrainingLauncher };