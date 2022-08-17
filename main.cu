/* Dieses Programm muss gelinkt werden gegen die Cublas library. Der Aufruf lautet "nvcc main.c -x cu -lcublas -o test und -g -G fuers Debuggen" */

/*  =======
    Include
    ======= */


/* Standard Bibliotheken */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

/* Cuda Bibliotheken */
#include <cuda_runtime.h>
#include "cublas_v2.h"
 
/*  =================
    Globale Variablen
    ================= */
int THREADS = 32;
int BLOCKS = 32;


/*  ==========
    Strukturen 
    ========== */

typedef struct {
    int number_of_layers;
    int* dimension_of_layer;
    float** weight_matrix;
} ff_network; /* ff=feedforward*/

typedef struct {
    int package_size;
	ff_network* network;
	float** adjacency_matrix;
	float** interim_result;
	float** delta_matrix;
	float** delta;
	float* lerngeschwindigkeit;
} trainer;

typedef struct {
	int number_of_inputs;
	float* input;
	float* output;
} training_data; 

/*  ================================================
    Elementweise Funktionen und Cuda Hilfsfunktionen
    ================================================ */
__global__
void apply_g_elementwise(int n, float* x, float* y)
{
    int i=0;
    for (i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
         y[i] = 1 / (1 + exp(-x[i]));
      }
}
__global__
void product_elementwise(int n, float* x, float* y)
{
    int i=0;
    for (i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
          y[i] = y[i]*x[i];
      }   
}
__global__
void apply_h_elementwise(int n, float* x, float* y)
{
    int i=0;
    for (i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n;
         i += blockDim.x * gridDim.x)
      {
          y[i] = x[i] - x[i]*x[i];
      }
}

/*  ===========================
    Funktionen fuers ff_network
    =========================== */
ff_network* create_and_init(int number_of_layers, int* dimensions)
{
    /*Jeder Pointer kriegt einen Speicherplatz, auf den er zeigt */
    ff_network* network = (ff_network*) malloc(sizeof(ff_network));
    network->number_of_layers = number_of_layers;
    network->dimension_of_layer = dimensions;
	network->weight_matrix = (float**) malloc(sizeof(float*)*(number_of_layers-1));

    /* Die Verbindungsmatrizen brauchen noch Speicher in der Grafikkarte. Zunaechst haben wir nur
    Speicher für einen Zeiger pro Matrix bekommen. Nun muessen wir diesen Zeiger pro Matrix auf jeweils eine Matrix in der GPU 
    zeigen lassen */
    cudaError_t cudaStatus;
    int i;
    for(i=0; i<network->number_of_layers-1; i++)
    {
        float* gpuPtr; // Nur fuer bessere Lesbarkeit
        int N = network->dimension_of_layer[i] * network->dimension_of_layer[i+1] * sizeof(float); //Nur fuer bessere Lesbarkeit
        cudaStatus = cudaMalloc((void**)&gpuPtr, N);
        if(cudaStatus != cudaSuccess)
        {
            printf("Speicherallokation fuer Matrix fehlgeschlagen.\n");
        }
        network->weight_matrix[i] = gpuPtr;
    }
    return network;
}

int set_weight_matrix(ff_network* network, int position, float* matrix)
{   
    cudaError_t cudaStatus;
    void* ziel_matrix = network->weight_matrix[position];
    int M = network->dimension_of_layer[position]; //Spalten
    int N = network->dimension_of_layer[position+1]; //Zeile
    cudaStatus = cudaMemcpy(ziel_matrix, matrix, N * M * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf ("Matrix konnte nicht in die Grafikkarte geschrieben werden.");
        return -1;
    }
	return 0;
}

int get_weight_matrix(ff_network* network, int position, float* matrix)
{	
    cudaError_t cudaStatus;
    void* start_matrix = network->weight_matrix[position];
    int M = network->dimension_of_layer[position]; //Spalten
    int N = network->dimension_of_layer[position+1]; //Zeile
    cudaStatus = cudaMemcpy(matrix, start_matrix, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf ("Matrix konnte nicht aus der Grafikkarte gelesen werden.");
        return -1;
    }
	return 0;
}

void calculate_function_onmany(cublasHandle_t handle, ff_network* network, int number_of_inputs, float* dev_input, float* output)
{
    int maxdim = 0;
    int i;
    float* tmp_vec;
    float* tmp_vec2;
    /* Maximale Dimension der Layer ermitteln: */
    for(i=0; i<network->number_of_layers; i++)
    {
        if(network->dimension_of_layer[i] > maxdim)
        {
            maxdim = network->dimension_of_layer[i];
        }
    }

   /*Speicherplatz auf der GPU fuer ZwischenErgebnis/ Output Vektor schaffen:*/
    cudaError_t cudaStat = cudaMalloc((void**)&tmp_vec, number_of_inputs*maxdim*sizeof(float));
    if (cudaStat != cudaSuccess) 
    {
        printf("Zwischenergebnis-Vektor konnte auf GPU nicht gespeichert werden.\n");
    }
    cudaStat = cudaMalloc((void**)&tmp_vec2, number_of_inputs* maxdim * sizeof(float));

    if (cudaStat != cudaSuccess) 
    {
        printf("Zweiter Zwischenergebnis-Vektor konnte auf GPU nicht gespeichert werden.\n");
    }
    for(i=0; i<number_of_inputs; i++)
    {
        cudaStat = cudaMemcpy(&(tmp_vec[i*maxdim]), &(dev_input[i*network->dimension_of_layer[0]]), network->dimension_of_layer[0]*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    if (cudaStat != cudaSuccess) 
    {
        printf ("Input konnte in der Grafikkarte nicht hin und her kopiert werden.");
    }


    float* ausgabe = (float*) malloc(sizeof(float)*number_of_inputs*maxdim);
    float alpha = 1.0f;
	float beta = 0.0f;
	for(i=0; i<network->number_of_layers-1; i++)
    { 
        cudaMemcpy(ausgabe, tmp_vec, maxdim*number_of_inputs*sizeof(float), cudaMemcpyDeviceToHost);
        
        cublasSgemm(handle,
                CUBLAS_OP_N,CUBLAS_OP_N,
                network->dimension_of_layer[i+1],
                number_of_inputs,
                network->dimension_of_layer[i],
                &alpha,
                network->weight_matrix[i],
                network->dimension_of_layer[i+1],
                tmp_vec, maxdim,
                &beta, tmp_vec2, maxdim
        );
        apply_g_elementwise<<<BLOCKS,THREADS>>>(maxdim*number_of_inputs, tmp_vec2, tmp_vec); 
    }
    cudaMemcpy(ausgabe,tmp_vec, maxdim*number_of_inputs*sizeof(float), cudaMemcpyDeviceToHost);
        
    int dim_lastlayer = network->dimension_of_layer[network->number_of_layers-1];
    for(i=0; i<number_of_inputs; i++)
    {
        cudaStat = cudaMemcpy(&(output[i*dim_lastlayer]), &(tmp_vec[i*maxdim]), dim_lastlayer*sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(tmp_vec);
    cudaFree(tmp_vec2);
}

/*  =======================
    Funktionen fuer Trainer
    ======================= */
trainer* initialize_trainer(int package_size, ff_network* network, float** adj_matrix)
{
    int i; // Schleifencounter
    int number_of_layers = network->number_of_layers; // Erhoehte Lesbarkeit

    trainer* ts = (trainer*) malloc(sizeof(trainer));
    
    ts->package_size = package_size;
    ts->adjacency_matrix = adj_matrix;
    ts->network = network;

    // Speicher fuer die Zwischenergebnisse allokieren
    ts->interim_result = (float**) malloc(sizeof(float*)*number_of_layers);
    for(i=0; i<number_of_layers; i++)
    {
        cudaMalloc((void**) &(ts->interim_result[i]), sizeof(float)*(network->dimension_of_layer[i])*package_size);
    } 
    
    // Speicher fuer die Deltas allokieren.
    ts->delta = (float**) malloc(sizeof(float*)*number_of_layers);
    for(i=0; i<number_of_layers; i++)
    {
        cudaMalloc((void**) &(ts->delta[i]), sizeof(float)*(network->dimension_of_layer[i])*package_size);
    } 
    // Speicher fuer die Delta-Matrizen allokieren
    ts->delta_matrix = (float**) malloc(sizeof(float*)*(number_of_layers-1));
    for(i=0; i<network->number_of_layers-1; i++)
    {
        cudaMalloc((void**) &(ts->delta_matrix[i]), sizeof(float)*(network->dimension_of_layer[i])*(network->dimension_of_layer[i+1]));
    } 
    return ts;
}

/*  =========================
    Funktionen fuers Training
    ========================= */
void adjust_weights(cublasHandle_t handle, training_data* tdata, trainer* ts, float learning_rate)
{
    float neg_alpha = -1.0f;
    float alpha = 1.0f;
	float beta = 0.0f;
    ff_network* network = ts->network;
    int networklength = network->number_of_layers;
	int number_of_inputs = tdata->number_of_inputs;

    ts->interim_result[0] = tdata->input;
    int i;
	for(i=0; i<network->number_of_layers-1; i++)
    { 
        cublasSgemm(handle,
                CUBLAS_OP_N,CUBLAS_OP_N,
                network->dimension_of_layer[i+1],
                number_of_inputs,
                network->dimension_of_layer[i],
                &alpha,
                network->weight_matrix[i],
                network->dimension_of_layer[i+1],
                ts->interim_result[i], network->dimension_of_layer[i],
                &beta, ts->interim_result[i+1], network->dimension_of_layer[i+1]
        );
        apply_g_elementwise<<<BLOCKS,THREADS>>>(network->dimension_of_layer[i+1]*number_of_inputs, ts->interim_result[i+1], ts->interim_result[i+1]);
    }
    apply_h_elementwise<<<BLOCKS,THREADS>>>(network->dimension_of_layer[networklength-1]*number_of_inputs, ts->interim_result[networklength-1], ts->delta[networklength-1]);
    cublasSaxpy(handle, network->dimension_of_layer[networklength-1]*number_of_inputs, &neg_alpha, tdata->output, 1, ts->interim_result[networklength-1], 1);
    product_elementwise<<<BLOCKS,THREADS>>>(network->dimension_of_layer[networklength-1]*number_of_inputs, ts->interim_result[networklength-1], ts->delta[networklength-1]);
    cublasSgemm(handle,
                CUBLAS_OP_N,CUBLAS_OP_T,
                network->dimension_of_layer[networklength-1],
                network->dimension_of_layer[networklength-2],
                number_of_inputs,
                &alpha,
                ts->delta[networklength-1],
                network->dimension_of_layer[networklength-1],
                ts->interim_result[networklength-2], network->dimension_of_layer[networklength-2],
                &beta, ts->delta_matrix[networklength-2], network->dimension_of_layer[networklength-1]
        );
        


    for(i=networklength-2; i>0; i--)
    {
        apply_h_elementwise<<<BLOCKS,THREADS>>>(network->dimension_of_layer[i]*number_of_inputs, ts->interim_result[i], ts->interim_result[i]);
        cublasSgemm(handle,
                CUBLAS_OP_T,CUBLAS_OP_N,
                network->dimension_of_layer[i], // Zeilen von A^t bzw. Spalten von A : Layer_i -> Layer_i+1
                number_of_inputs,
                network->dimension_of_layer[i+1],
                &alpha,
                network->weight_matrix[i],
                network->dimension_of_layer[i+1],
                ts->delta[i+1], network->dimension_of_layer[i+1],
                &beta, ts->delta[i], network->dimension_of_layer[i]
        );
        product_elementwise<<<BLOCKS,THREADS>>>(network->dimension_of_layer[i]*number_of_inputs, ts->interim_result[i], ts->delta[i]);
        cublasSgemm(handle,
                CUBLAS_OP_N,CUBLAS_OP_T,
                network->dimension_of_layer[i], // m 
                network->dimension_of_layer[i-1], // n
                number_of_inputs, // k
                &alpha, // alpha
                ts->delta[i], // A
                network->dimension_of_layer[i], //lda
                ts->interim_result[i-1], network->dimension_of_layer[i-1], //B,ldb
                &beta, ts->delta_matrix[i-1], network->dimension_of_layer[i] // beta,C,ldc
        );
    }

    for(i=0; i<networklength-1; i++)
    {   
        product_elementwise<<<BLOCKS,THREADS>>>(network->dimension_of_layer[i]*network->dimension_of_layer[i+1], ts->adjacency_matrix[i], ts->delta_matrix[i]);      
        float a = neg_alpha * learning_rate; //ts->lerngeschwindigkeit[i];
        cublasSaxpy(handle, network->dimension_of_layer[networklength-1]*number_of_inputs, &a, ts->delta_matrix[i], 1, network->weight_matrix[i], 1);
    }
}

void train_network(cublasHandle_t handle, trainer* ts, training_data* tdata, int number_packages, int number_iteration, float learning_rate)
{
        ff_network* network = ts->network;
        training_data* current_training_data = (training_data*) malloc(sizeof(training_data));
        current_training_data->number_of_inputs = ts->package_size;
        
        int i;
        int counter_in = 0;
        int counter_out = 0;

        for(i=0; i<number_packages; i++)
        {
            current_training_data->input = &(tdata->input[counter_in]);
            current_training_data->output = &(tdata->output[counter_out]);
            counter_in += (ts->package_size)*(network->dimension_of_layer[0]);
            counter_out += (ts->package_size)*(network->dimension_of_layer[network->number_of_layers-1]);
            int j = 0;
            for(j=0; j<number_iteration; j++)
            {
                adjust_weights(handle, current_training_data, ts, learning_rate);
            }
            
        }
        free(current_training_data);
}

void shuffle_training_data(training_data* tdata, int input_dimension, int output_dimension)
{
	srand(time(NULL)); // Seeding der Zufallszahlen

	int input_size = tdata->number_of_inputs;
	float* input = tdata->input;
	float* output = tdata->output;

 	int i = 0;
	float* dev_input;
	float* dev_output;

	cudaMalloc((void**) &dev_input, input_dimension * sizeof(float));
  cudaMalloc((void**) &dev_output, output_dimension * sizeof(float));

	for(i = 0; i < input_size; i++)
	{
		int current_end = input_size-i;
		/*
    int r; 
    while((r = rand()) > (RAND_MAX-(RAND_MAX % current_end))); //uniform-verteilt ueber 0-current-end
    r = r % current_end;
    */
    int r = rand() % current_end;
		/* Die r-te Stelle wird in den Zwischenspeicher kopiert */
		cudaMemcpy(dev_input, input+r*input_dimension, input_dimension * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_output, output+r*output_dimension, output_dimension * sizeof(float), cudaMemcpyDeviceToDevice);

		/* Die letzte noch nicht bearbeitete Stelle wird an die r-te Stelle kopiert.*/
		cudaMemcpy(input+r*input_dimension, input+(current_end-1)*input_dimension, input_dimension * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(output+r*output_dimension, output+(current_end-1)*output_dimension, output_dimension * sizeof(float), cudaMemcpyDeviceToDevice);
		
		/* Der Zwischenspeicher wird an die letzte nicht bearbeitete Stelle kopiert.*/
		cudaMemcpy(input+(current_end-1)*input_dimension, dev_input, input_dimension * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(output+(current_end-1)*output_dimension, dev_output, output_dimension * sizeof(float), cudaMemcpyDeviceToDevice);	
	
	}
	cudaFree(dev_input);
	cudaFree(dev_output);
} 

/*  ==================================================
    Funktionen um Daten in eine Textdatei zu schreiben
    ================================================== */
void save_output_to_file(const char *filename, float* output, int outputdimension, int number_of_outputs)
{
    FILE* file = fopen(filename, "w");
    fprintf(file,"%d %d ", outputdimension, number_of_outputs);
    int endsize = (outputdimension+1)*number_of_outputs;

    int i;
    for(i=0; i<endsize; i++)
    {
        if(i%(outputdimension+1) != 0 )
        {
            fprintf(file,"%f ", output[i]);
        }
    }
}

void save_network_to_file(ff_network* network, const char* filename)
{
    //Datei erzeugen lassen.
    FILE* file = fopen(filename, "w");
    
    fprintf(file,"%d ", network->number_of_layers);

    int i;
    for(i=0; i<network->number_of_layers; i++)
    {
        fprintf(file,"%d ", network->dimension_of_layer[i]);
    }

    for(i=0; i<network->number_of_layers-1; i++)
    {
        int k = 0;
        int N = network->dimension_of_layer[i];
        int M = network->dimension_of_layer[i+1];
        int size = N*M;
        float* devPtrA = network->weight_matrix[i];
	    float* matrix = (float*) malloc(sizeof(float)*size);
	    cublasGetMatrix (M, N, sizeof(*matrix), devPtrA, M, matrix, M);// Muss noch ausgetauscht werden....
        for(k=0;k<size; k++)
        {
            fprintf(file, "%.16f ", matrix[k]);
        }
        free(matrix);   
    }
    fclose(file);
}
/*  ==========================================
    Funktionen fuer Daten aus einem File lesen
    ========================================== */
void load_input_from_file_to_dev(const char* filename, float** input, int* number)
{
    FILE* file = fopen(filename, "r");
    int inputdimension;
    int number_of_inputs;
    int i;

    fscanf(file, "%d", &inputdimension);
    fscanf(file, "%d", &number_of_inputs);

    *number = number_of_inputs;
    int endsize = (inputdimension+1)*number_of_inputs;
    float* hostinput = (float*) malloc(endsize*sizeof(float));

    for(i=0; i<endsize; i++)
    {
        if(i%(inputdimension+1) == 0)
        {
            hostinput[i] = 0.5f;
        }else{
            fscanf(file, "%f", &hostinput[i]);
        }
    }
    
    
    float* devinput;
    cudaError_t cudaStatus = cudaMalloc((void**)&devinput, endsize*sizeof(float));
    if(cudaStatus != cudaSuccess)
    {
        printf("Speicherallokation fuer Matrix fehlgeschlagen.\n");
    }

    cudaStatus = cudaMemcpy(devinput, hostinput, endsize*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf ("Input konnte nicht in die Grafikkarte geschrieben werden.");
    }
    free(hostinput);
    *input = devinput;
}

ff_network* load_network_from_file(const char *filename)
{
    FILE* f = fopen(filename, "r");
    if (f == NULL)
    {
        printf("Datei konnte nicht geoeffnet werden!");
    }
    int number_of_layers;
    fscanf(f, " %d", &number_of_layers); // "%d" liest die Zahlen (d=decimal) bis zum naechsten
    int* dimensions = (int*) malloc(number_of_layers*sizeof(int));
    int i=0;
    for(i=0; i<number_of_layers; i++)
    {
        fscanf(f, " %d", &dimensions[i]);
    }
    ff_network* network = create_and_init(number_of_layers,dimensions);
    
    for(i=0; i<number_of_layers-1; i++)
    {
        int size = dimensions[i]*dimensions[i+1];
        float* matrix=(float*) malloc(size*sizeof(float));
        int k=0;
        for(k=0; k<size; k++)
        {
            fscanf(f, "%f", &matrix[k]);
        }
        set_weight_matrix(network,i,matrix); 
        free(matrix);    
    }
    fclose(f);
    return network;
}

float** load_adjacency_matrix_from_file(const char* filename)
{
    ff_network* tmp_network = load_network_from_file(filename);
    float** result = tmp_network->weight_matrix;

    free(tmp_network->dimension_of_layer);
    free(tmp_network);

    return result;
}

training_data* load_training_data_from_file(const char* filename)
{
    FILE* file = fopen(filename, "r");
    int inputdimension;
    int outputdimension;
    int number_of_inputs;
    int i;

    fscanf(file, "%d", &inputdimension);
    fscanf(file, "%d", &outputdimension);
    fscanf(file, "%d", &number_of_inputs);

    float* input = (float*) malloc((inputdimension+1)*number_of_inputs*sizeof(float));
    for(i=0; i<(inputdimension+1)*number_of_inputs; i++)
    {
        if(i%(inputdimension+1) == 0){
            input[i] = 0.5f;
        }else{
            fscanf(file, "%f", &input[i]);
        }
        
    }

    float* output = (float*) malloc((outputdimension+1)*number_of_inputs*sizeof(float));
    for(i=0; i<(outputdimension+1)*number_of_inputs; i++)
    {
        if(i%(outputdimension+1) == 0){
            output[i] = 0.5f;
        }else{
            fscanf(file, "%f", &output[i]);
        }
        
    }
    

    training_data* tdata = (training_data*) malloc(sizeof(training_data));
    tdata->number_of_inputs = number_of_inputs;

    float* dev_input;
    float* dev_output;

    cudaMalloc((void**) &dev_input, (inputdimension+1)*number_of_inputs*sizeof(float));
    cudaMalloc((void**) &dev_output, (outputdimension+1)*number_of_inputs*sizeof(float));

    cudaMemcpy(dev_input, input, (inputdimension+1)*number_of_inputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_output, output, (outputdimension+1)*number_of_inputs*sizeof(float), cudaMemcpyHostToDevice);
    
    free(input);
    free(output);
    fclose(file);

    tdata->input = dev_input;
    tdata->output = dev_output;

    return tdata;
}
/*  =====================================
    Funktionen fuer CommandLine-Usibility
    ===================================== */
void print_help()
{
    printf("HILFE\n*****\n\nPARAMETER\t\t\t BESCHREIBUNG\n\t-a\t\t (A)ufteilung der Cuda Kerne z.B. -a 32x64 entspricht 32 Bloecke auf 64 Threads.\n\n\t-c\t\t (C)alculieren. Gibt an, ob das Netzwerk am Ende eine Berechnung durchfuehren soll.\n\n\t-d\t\t (D)aten. Gibt den Pfad an, wo die Datendatei hinterlegt ist z.B -d Netzwerk/datenfile.txt.\n\n\t-h\t\t (H)ilfe. Zeigt diese Hilfe an :D.\n\n\t-i\t\t (I)nput. Gibt den Pfad an, wo die Inputdatei hinterlegt ist z.B -d Netzwerk/input.txt.\n\n\t-j\t\t Ad(j)acency. Gibt den Pfad an, wo die Adjacency-Matrix hinterlegt ist z.B -j Netzwerk/adjacecy.txt.\n\n\t-l\t\t (L)ernen. Gibt an, ob das Netzwerk trainiert werden soll.\n\n\t-n\t\t (N)etzwerk. Gibt den Pfad an, wo das Netzwerk hinterlegt ist z.B -d Netzwerk/netzwerk.txt.\n\n\t-o\t\t (O)utput. Gibt den Pfad an, wo die Outputdatei hinterlegt ist z.B -d Netzwerk/output.txt.\n\n\t-p\t\t (P)ackage-Size. Hier kann die Paketgroesse geaendert werden z.B. -p 100.\n\n\t-s\t\t (S)peichern. Gibt den Pfad an, wo das Netzwerk gespeichert werden soll z.B -d Netzwerk/save.txt.\n\n\t-t\t\t (T)rainer. Hier kann man den Trainer einstellen z.B -t ANZAHLPAKETExWIEDERHOLUNGxLERNRATE.\n\n\t-v\t\t (V)erbose. Gibt Zwischenschritte in der Console aus.\n\n\n\n");
}


int opterr = 1;             /* if error message should be printed */
int optind = 1;             /* index into parent argv vector */
int optopt;                 /* character checked for validity */
int optreset = 0;               /* reset getopt */
const char* optarg;               /* argument associated with option */

#define BADCH   (int)'?'
#define BADARG  (int)':'
#define EMSG    ""

int getopt(int nargc, char* const nargv[], const char* ostr)
{
    static const char* place = EMSG;              /* option letter processing */
    const char* oli;                        /* option letter list index */

    if (optreset || !*place) 
    {              /* update scanning pointer */
        optreset = 0;
        if (optind >= nargc || *(place = nargv[optind]) != '-') 
        {
            place = EMSG;
            return (-1);
        }
        if (place[1] && *++place == '-') 
        {      /* found "--" */
            ++optind;
            place = EMSG;
            return (-1);
        }
    }                                       /* option letter okay? */
    if ((optopt = (int)*place++) == (int)':' || !(oli = strchr(ostr, optopt))) 
    {
      /*
      * if the user didn't specify '-' as an option,
      * assume it means -1.
      */
        if (optopt == (int)'-') return (-1);
        if (!*place) ++optind;
        if (opterr && *ostr != ':') (void)printf("illegal option -- %c\n", optopt);
      return (BADCH);
    }
    if (*++oli != ':') 
    {                    /* don't need argument */
        optarg = NULL;
        if (!*place) ++optind;
    }
    else
    {                                  /* need an argument */
        if (*place)                     /* no white space */
            optarg = place;
        else if (nargc <= ++optind) 
        {   /* no arg */
            place = EMSG;
            if (*ostr == ':')
            return (BADARG);
            if (opterr)
            (void) printf("option requires an argument -- %c\n", optopt);
            return (BADCH);
        }
        else                            /* white space */
        optarg = nargv[optind];
        place = EMSG;
        ++optind;
    }
    return (optopt);                        /* dump back option letter */
}
/*  =============
    Main-Funktion
    ============= */
int main(int argc, char **argv)
{	
    int c; // Variable fuer Getopt abzuarbeiten.
	char a_value[50]; //Opt: -a Blocks x Threads
	char* a_number_of_blocks;
	char* a_number_of_threads;
    const char* output_file = "Netzwerk/output.txt"; //Opt: -o Outputfile
    const char* network_file = "Netzwerk/network.txt"; // Opt: -n Netzwerk, dass geladen werden soll
    const char* input_file = "Netzwerk/input.txt"; //Opt -i inputfile
    int help_call = 0;
    const char* trainingsdata_file = "Netzwerk/trainingsdata.txt"; //Opt: -d Trainingsdatenfile
    int s_value = 0; //Opt: -s  Boolean if Network should be saved.
    const char* save_network_file; 
    int p_package_size = 100; //Opt: -p Falls die package size geaendert werden soll
    const char* adj_matrix_file = "Netzwerk/adj_matrix.txt";   //Opt: -j adjacency matrix
    char t_value[50];             //Opt: -t Konfiguriert den Trainer
    char *t_number_packages_string;
    char *t_number_iterations_string;
    char *t_learning_rate_string;
    int t_number_packages = 2000;
    int t_number_iterations = 1;
    float t_learning_rate = 0.5f;
    int c_calculate = 0; //Opt: -c  Boolean if Network should calculate
    int l_learn = 0; //Opt: -l Boolean if Network should learn
    int verbose = 0; //Opt: -v Boolean for verbose mode.

	while((c = getopt(argc, argv, "a:d:i:j:n:o:p:t:s:chlv")) != -1)
	{
		switch(c)
		{
		case 'a':
			strcpy(a_value, optarg);
			a_number_of_blocks = strtok(a_value, "x");
			a_number_of_threads = strtok(NULL, "x");
			BLOCKS = atoi(a_number_of_blocks);
			THREADS = atoi(a_number_of_threads);
			break;
        case 'c':
            c_calculate = 1;
            break;
        case 'd':
            trainingsdata_file = optarg;
            break;
        case 'h':
            print_help();
            help_call = 1;
            break;
        case 'i':
            input_file = optarg;
            break;
        case 'j':
            adj_matrix_file = optarg;
            break;
        case 'l':
            l_learn = 1;
            break;
        case 'n':
            network_file = optarg;
            break;
        case 'o':
            output_file = optarg;
            break;
        case 'p':
            p_package_size = atoi(optarg);
            break;
        case 't':
            strcpy(t_value,optarg);
            t_number_packages_string = strtok(t_value, "x");
            t_number_iterations_string = strtok(NULL, "x");
            t_learning_rate_string = strtok(NULL, "x");
	        t_number_packages = atoi(t_number_packages_string);
            t_number_iterations = atoi(t_number_iterations_string);
            t_learning_rate = atof(t_learning_rate_string);
			break;
        case 's':
            s_value = 1;
            save_network_file = optarg;
            break;
        case 'v':
            verbose = 1;
            break;
		case '?':
			if(optopt=='a') fprintf(stderr, "Option -%c requires an argument\n",optopt);
			else if (isprint(optopt)) fprintf(stderr, "Unkown option\n");
			else fprintf(stderr, "Unkown option character\n");
			break;
		}
	}

    if(optind < argc) 
    {
        help_call = 1;
        printf("Es sind unbekannte Parameter aufgerufen worden. Benutze den Parameter -h fuer eine Liste aller Parameter.");
    }

    if(help_call != 0) return 0;

    
    if(verbose != 0) printf("Lade Netzwerk aus Datei: %s\n",network_file);
    ff_network* network = load_network_from_file(network_file);
	
    if(verbose != 0) printf("Netzwerk geladen. Erzeuge eine CuBLAS-Instanz.\n");
    cublasHandle_t handle;
    cublasCreate(&handle);

    if(l_learn != 0)
    {
        if(verbose != 0) printf("Lade Adjazenmatrix aus Datei:%s\n",adj_matrix_file);
        float** adj_matrix = load_adjacency_matrix_from_file(adj_matrix_file);

        if(verbose != 0) printf("Adjazenmatrix geladen. Initialisiere die Trainingsdaten und den Trainer.\n");
        training_data* td = load_training_data_from_file(trainingsdata_file);
        trainer* ts = initialize_trainer(p_package_size, network, adj_matrix);

        if(verbose != 0) printf("Trainer und Trainingsdaten initialisiert. Starte das Training.\n");
        float size_of_training_data = (float) td->number_of_inputs;
        float number_of_data= (float) (t_number_packages*p_package_size);
        int number_of_iterations = ((int) (number_of_data/size_of_training_data));
        int max_number_of_packages_per_iteration = (int) (size_of_training_data/ ((float) p_package_size));
        int input_dimension = network->dimension_of_layer[0]-1;
        int output_dimension = network->dimension_of_layer[network->number_of_layers-1]-1;
        int i;
        for(i = 0; i < number_of_iterations; i++)
        {
            if(verbose != 0) printf("%d/%d\n",i+1,number_of_iterations);
            shuffle_training_data(td, input_dimension+1, output_dimension+1); //+1 wegen des Biases
            train_network(handle, ts , td, max_number_of_packages_per_iteration, t_number_iterations, t_learning_rate);
        }
        int residual_package_number = t_number_packages - max_number_of_packages_per_iteration*number_of_iterations;
        shuffle_training_data(td, input_dimension, output_dimension);
        train_network(handle, ts , td, residual_package_number, t_number_iterations, t_learning_rate);
        if(verbose != 0) printf("Training wurde beendet.");
    }

    if(c_calculate !=0)
    {
        if(verbose != 0) printf("Lade Input aus Datei: %s\n",input_file);
    int number_of_inputs;
	float* dev_input;
    load_input_from_file_to_dev(input_file, &dev_input, &number_of_inputs);
        if(verbose != 0) printf("Starte Berechnung.\n");
        float* output=(float*) malloc(number_of_inputs*network->dimension_of_layer[network->number_of_layers-1]*sizeof(float));
        calculate_function_onmany(handle, network, number_of_inputs,  dev_input,  output);
	if(verbose != 0) printf("Schreibe Output.\n");
        save_output_to_file(output_file, output, network->dimension_of_layer[network->number_of_layers-1]-1, number_of_inputs); 
        free(output);
    }
    
    if(s_value != 0)
    {
        if(verbose != 0) printf("Netzwerk wird gespeichert.\n");
        save_network_to_file(network, save_network_file);
    }
    return 0;
}
