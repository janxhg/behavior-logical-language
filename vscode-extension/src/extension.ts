import * as vscode from 'vscode';
import * as path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';

const execAsync = promisify(exec);

interface ValidationResult {
    line: number;
    column: number;
    severity: 'error' | 'warning' | 'info';
    message: string;
    rule?: string;
}

class BrainLLLanguageServer {
    private readonly diagnosticCollection: vscode.DiagnosticCollection;
    private validatorPath: string;
    private outputChannel: vscode.OutputChannel;

    constructor() {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('brainll');
        this.outputChannel = vscode.window.createOutputChannel('BrainLL');
        this.validatorPath = this.getValidatorPath();
    }

    private getValidatorPath(): string {
        const config = vscode.workspace.getConfiguration('brainll');
        const customPath = config.get<string>('validatorPath');
        
        if (customPath && customPath.trim() !== '') {
            return customPath;
        }

        // Try to find validator in common locations
        const commonPaths = [
            'brainll_validator',
            'brainll_validator.exe',
            path.join(vscode.workspace.rootPath || '', 'build', 'brainll_validator'),
            path.join(vscode.workspace.rootPath || '', 'build', 'brainll_validator.exe'),
            path.join(vscode.workspace.rootPath || '', 'bin', 'brainll_validator'),
            path.join(vscode.workspace.rootPath || '', 'bin', 'brainll_validator.exe')
        ];

        return commonPaths[0]; // Default to system PATH
    }

    async validateDocument(document: vscode.TextDocument): Promise<void> {
        if (document.languageId !== 'brainll') {
            return;
        }

        const config = vscode.workspace.getConfiguration('brainll');
        if (!config.get<boolean>('validation.enabled', true)) {
            return;
        }

        try {
            const tempFile = path.join(path.dirname(__filename), 'temp_' + Date.now() + '.bll');
            fs.writeFileSync(tempFile, document.getText());

            const strictMode = config.get<boolean>('validation.strictMode', false);
            const command = `"${this.validatorPath}" "${tempFile}" --json ${strictMode ? '--strict' : ''}`;
            
            const { stdout, stderr } = await execAsync(command);
            
            // Clean up temp file
            fs.unlinkSync(tempFile);

            if (stderr) {
                this.outputChannel.appendLine(`Validator stderr: ${stderr}`);
            }

            const results: ValidationResult[] = stdout ? JSON.parse(stdout) : [];
            this.updateDiagnostics(document, results);

        } catch (error) {
            this.outputChannel.appendLine(`Validation error: ${error}`);
            // Clear diagnostics on error
            this.diagnosticCollection.set(document.uri, []);
        }
    }

    private updateDiagnostics(document: vscode.TextDocument, results: ValidationResult[]): void {
        const diagnostics: vscode.Diagnostic[] = results.map(result => {
            const line = Math.max(0, result.line - 1); // Convert to 0-based
            const column = Math.max(0, result.column - 1);
            const range = new vscode.Range(line, column, line, column + 10);
            
            let severity: vscode.DiagnosticSeverity;
            switch (result.severity) {
                case 'error':
                    severity = vscode.DiagnosticSeverity.Error;
                    break;
                case 'warning':
                    severity = vscode.DiagnosticSeverity.Warning;
                    break;
                default:
                    severity = vscode.DiagnosticSeverity.Information;
            }

            const diagnostic = new vscode.Diagnostic(range, result.message, severity);
            diagnostic.source = 'BrainLL';
            if (result.rule) {
                diagnostic.code = result.rule;
            }

            return diagnostic;
        });

        this.diagnosticCollection.set(document.uri, diagnostics);
    }

    dispose(): void {
        this.diagnosticCollection.dispose();
        this.outputChannel.dispose();
    }
}

class BrainLLCompletionProvider implements vscode.CompletionItemProvider {
    private keywords = [
        'region', 'population', 'connection', 'plasticity', 'learning', 'simulation',
        'import', 'export', 'include', 'config'
    ];

    private neuronTypes = [
        'LIF', 'Izhikevich', 'HodgkinHuxley', 'LSTM', 'Attention', 'Adaptive', 'Transformer'
    ];

    private plasticityTypes = [
        'STDP', 'Hebbian', 'AntiHebbian', 'BCM', 'Oja'
    ];

    private learningTypes = [
        'Supervised', 'Reinforcement', 'Unsupervised'
    ];

    private parameters = [
        'size', 'threshold', 'decay', 'learning_rate', 'weight', 'delay', 'probability',
        'from', 'to', 'type', 'plasticity', 'neuron_type', 'activation', 'dimensions',
        'tau_m', 'tau_s', 'v_rest', 'v_reset', 'v_thresh', 'refractory_period',
        'a', 'b', 'c', 'd', 'g_na', 'g_k', 'g_l', 'e_na', 'e_k', 'e_l', 'c_m',
        'input_size', 'hidden_size', 'output_size', 'num_heads', 'dropout', 'ff_dim',
        'num_layers', 'adaptation_rate', 'threshold_adaptation', 'spike_adaptation'
    ];

    provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken,
        context: vscode.CompletionContext
    ): vscode.ProviderResult<vscode.CompletionItem[] | vscode.CompletionList> {
        const config = vscode.workspace.getConfiguration('brainll');
        if (!config.get<boolean>('autocompletion.enabled', true)) {
            return [];
        }

        const lineText = document.lineAt(position).text;
        const beforeCursor = lineText.substring(0, position.character);

        const completions: vscode.CompletionItem[] = [];

        // Context-aware completions
        if (beforeCursor.includes('neuron_type:')) {
            completions.push(...this.createCompletions(this.neuronTypes, vscode.CompletionItemKind.Enum));
        } else if (beforeCursor.includes('plasticity:') || beforeCursor.includes('type:')) {
            completions.push(...this.createCompletions(this.plasticityTypes, vscode.CompletionItemKind.Enum));
            completions.push(...this.createCompletions(this.learningTypes, vscode.CompletionItemKind.Enum));
        } else {
            // General completions
            completions.push(...this.createCompletions(this.keywords, vscode.CompletionItemKind.Keyword));
            completions.push(...this.createCompletions(this.parameters, vscode.CompletionItemKind.Property));
        }

        // Add population references
        const populations = this.extractPopulations(document);
        completions.push(...populations.map(pop => {
            const item = new vscode.CompletionItem(pop, vscode.CompletionItemKind.Reference);
            item.detail = 'Population reference';
            return item;
        }));

        return completions;
    }

    private createCompletions(items: string[], kind: vscode.CompletionItemKind): vscode.CompletionItem[] {
        return items.map(item => new vscode.CompletionItem(item, kind));
    }

    private extractPopulations(document: vscode.TextDocument): string[] {
        const populations: string[] = [];
        const text = document.getText();
        const populationRegex = /(?:^|\n)\s*population\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{/g;
        const regionRegex = /(?:^|\n)\s*region\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{[^}]*population\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{/g;
        
        let match;
        while ((match = populationRegex.exec(text)) !== null) {
            populations.push(match[1]);
        }
        
        while ((match = regionRegex.exec(text)) !== null) {
            populations.push(`${match[1]}.${match[2]}`);
        }
        
        return populations;
    }
}

class BrainLLHoverProvider implements vscode.HoverProvider {
    private documentation: { [key: string]: string } = {
        'LIF': 'Leaky Integrate-and-Fire neuron model. Simple but efficient model with threshold-based spiking.',
        'Izhikevich': 'Izhikevich neuron model. Computationally efficient model that can reproduce various firing patterns.',
        'HodgkinHuxley': 'Hodgkin-Huxley neuron model. Biologically detailed model with sodium and potassium channels.',
        'LSTM': 'Long Short-Term Memory neuron. Recurrent neural network unit with gating mechanisms.',
        'Attention': 'Multi-head attention mechanism. Transformer-based attention for sequence processing.',
        'Adaptive': 'Adaptive neuron with dynamic threshold and spike adaptation.',
        'Transformer': 'Full transformer neuron with self-attention and feed-forward layers.',
        'STDP': 'Spike-Timing Dependent Plasticity. Synaptic strength changes based on spike timing.',
        'Hebbian': 'Hebbian learning rule. "Neurons that fire together, wire together."',
        'region': 'Defines a brain region containing populations and connections.',
        'population': 'Defines a group of neurons with shared properties.',
        'connection': 'Defines synaptic connections between populations.',
        'size': 'Number of neurons in the population.',
        'threshold': 'Firing threshold for the neuron.',
        'weight': 'Synaptic weight strength.',
        'delay': 'Synaptic transmission delay in timesteps.'
    };

    provideHover(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken
    ): vscode.ProviderResult<vscode.Hover> {
        const range = document.getWordRangeAtPosition(position);
        if (!range) {
            return;
        }

        const word = document.getText(range);
        const documentation = this.documentation[word];
        
        if (documentation) {
            return new vscode.Hover(new vscode.MarkdownString(documentation));
        }
    }
}

export function activate(context: vscode.ExtensionContext) {
    const languageServer = new BrainLLLanguageServer();
    const completionProvider = new BrainLLCompletionProvider();
    const hoverProvider = new BrainLLHoverProvider();

    // Register providers
    context.subscriptions.push(
        vscode.languages.registerCompletionItemProvider('brainll', completionProvider, ':', '.'),
        vscode.languages.registerHoverProvider('brainll', hoverProvider)
    );

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('brainll.validateSyntax', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'brainll') {
                await languageServer.validateDocument(editor.document);
                vscode.window.showInformationMessage('BrainLL syntax validation completed.');
            }
        }),

        vscode.commands.registerCommand('brainll.formatDocument', () => {
            vscode.window.showInformationMessage('BrainLL formatting is not yet implemented.');
        }),

        vscode.commands.registerCommand('brainll.showDocumentation', () => {
            vscode.env.openExternal(vscode.Uri.parse('https://github.com/brainll-project/docs'));
        })
    );

    // Auto-validation on document changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument(async (event: vscode.TextDocumentChangeEvent) => {
            if (event.document.languageId === 'brainll') {
                // Debounce validation
                setTimeout(() => {
                    languageServer.validateDocument(event.document);
                }, 500);
            }
        }),

        vscode.workspace.onDidOpenTextDocument(async (document: vscode.TextDocument) => {
            if (document.languageId === 'brainll') {
                await languageServer.validateDocument(document);
            }
        })
    );

    context.subscriptions.push(languageServer);

    vscode.window.showInformationMessage('BrainLL Language Support activated!');
}

export function deactivate() {
    // Cleanup is handled by context.subscriptions
}