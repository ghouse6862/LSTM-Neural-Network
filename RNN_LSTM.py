import numpy as np           
import pandas as pd

class Model:
    
    def __init__(self,train,iterations):
        raw_input = pd.read_csv(train)
        self.df = self.pre_process(raw_input)
        self.v = self.create_vocabulary()
        self.vlength = len(self.v)
        self.char_id,self.id_char = self.create_characterids()
        self.train_df = self.create_trainset()
        self.input_units = 100
        self.hidden_units = 256
        self.output_units = self.vlength
        self.learning_rate = 0.01
        self.b1 = 0.8
        self.b2 = 0.9
        self.p = self.initialize_parameters()
        self.epochs = iterations
        
        
    def pre_process(self,raw_input):
        raw_input = np.array(raw_input['Name'][:25000]).reshape(-1,1)
        raw_input = [n.lower() for n in raw_input[:,0]]
        raw_input = np.array(raw_input).reshape(-1,1)
        
        max_len = 0
        i = 0
        while (i<len(raw_input)): 
            max_len = max(max_len,len(raw_input[i,0]))
            i = i+1
        
        x = 0
        while(x < len(raw_input)):
            length = (max_len - len(raw_input[x,0]))
            s = ' '*length
            raw_input[x,0] = ''.join([raw_input[x,0],s])
            x = x+1
        
        return raw_input
    
    def create_vocabulary(self):
        vocabulary = list()
        for name in self.df[:,0]:
            vocabulary.extend(list(name))

        return set(vocabulary)
    
    def create_characterids(self):
        c_id = dict()
        id_c = dict()
        for i,c in enumerate(self.v):
            c_id[c] = i
            id_c[i] = c
        
        return c_id,id_c
    
    def create_trainset(self):
        training_data = []
        b_len = 20

        for i in range(len(self.df)-b_len+1):
            s = i*b_len
            e = s+b_len  
            b_df = self.df[s:e]
            if(len(b_df)!=b_len):
                break
        
            char_list = []
            for k in range(len(b_df[0][0])):
                b_dataset = np.zeros([b_len,self.vlength])
                for j in range(b_len):
                    name = b_df[j][0]
                    char_index = self.char_id[name[k]]
                    b_dataset[j,char_index] = 1.0
            
                char_list.append(b_dataset)
            
            training_data.append(char_list)
        
        return training_data
    
    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def tanh_activation(self,X):
        return np.tanh(X)

    def softmax(self,X):
        sf = np.exp(X)/np.sum(np.exp(X),axis=1).reshape(-1,1)
        return sf

    def tanh_derivative(self,X):
        return 1-(X**2)
    
    def initialize_V(self):
        Vigw = np.zeros(self.p['igw'].shape)
        Vogw = np.zeros(self.p['ogw'].shape)
        Vhow = np.zeros(self.p['how'].shape)
        Vfgw = np.zeros(self.p['fgw'].shape)
        Vggw = np.zeros(self.p['ggw'].shape)
    
        V = dict()
        V['vigw'] = Vigw
        V['vogw'] = Vogw
        V['vhow'] = Vhow
        V['vfgw'] = Vfgw
        V['vggw'] = Vggw
        
        return V

    def initialize_S(self):
        Sigw = np.zeros(self.p['igw'].shape)
        Sogw = np.zeros(self.p['ogw'].shape)
        Show = np.zeros(self.p['how'].shape)
        Sfgw = np.zeros(self.p['fgw'].shape)
        Sggw = np.zeros(self.p['ggw'].shape)
    
        S = dict()
        S['sigw'] = Sigw
        S['sogw'] = Sogw
        S['show'] = Show
        S['sfgw'] = Sfgw
        S['sggw'] = Sggw
        
        return S
    
    def initialize_parameters(self):
        ig_wts  = np.random.normal(0,0.1,(self.input_units+self.hidden_units,self.hidden_units))
        og_wts = np.random.normal(0,0.1,(self.input_units+self.hidden_units,self.hidden_units))
        ho_wts = np.random.normal(0,0.1,(self.hidden_units,self.output_units))
        fgt_wts = np.random.normal(0,0.1,(self.input_units+self.hidden_units,self.hidden_units))
        gt_wts   = np.random.normal(0,0.1,(self.input_units+self.hidden_units,self.hidden_units))
    
        pms = dict()
        pms['igw'] = ig_wts
        pms['ogw'] = og_wts
        pms['how'] = ho_wts
        pms['fgw'] = fgt_wts
        pms['ggw'] = gt_wts
    
        return pms
    
    def update_parameters(self,dts,V,S,t):
        digw = dts['digw']
        dogw = dts['dogw']
        dhow = dts['dhow']
        dfgw = dts['dfgw']
        dggw = dts['dggw']
    
        igw = self.p['igw']
        ogw = self.p['ogw']
        how = self.p['how']
        fgw = self.p['fgw']
        ggw = self.p['ggw']
    
        vigw = V['vigw']
        vogw = V['vogw']
        vhow = V['vhow']
        vfgw = V['vfgw']
        vggw = V['vggw']
        
        sigw = S['sigw']
        sogw = S['sogw']
        show = S['show']
        sfgw = S['sfgw']
        sggw = S['sggw']
    
        vigw = (self.b1*vigw + (1-self.b1)*digw)
        vogw = (self.b1*vogw + (1-self.b1)*dogw)
        vhow = (self.b1*vhow + (1-self.b1)*dhow)
        vfgw = (self.b1*vfgw + (1-self.b1)*dfgw)
        vggw = (self.b1*vggw + (1-self.b1)*dggw)
    
        sigw = (self.b2*sigw + (1-self.b2)*(digw**2))
        sogw = (self.b2*sogw + (1-self.b2)*(dogw**2))
        show = (self.b2*show + (1-self.b2)*(dhow**2))
        sfgw = (self.b2*sfgw + (1-self.b2)*(dfgw**2))
        sggw = (self.b2*sggw + (1-self.b2)*(dggw**2))
    
        igw = igw - self.learning_rate*((vigw)/(np.sqrt(sigw) + 1e-6))
        ogw = ogw - self.learning_rate*((vogw)/(np.sqrt(sogw) + 1e-6))
        how = how - self.learning_rate*((vhow)/(np.sqrt(show) + 1e-6))
        fgw = fgw - self.learning_rate*((vfgw)/(np.sqrt(sfgw) + 1e-6))
        ggw = ggw - self.learning_rate*((vggw)/(np.sqrt(sggw) + 1e-6))
    
        self.p['igw'] = igw
        self.p['ogw'] = ogw
        self.p['how'] = how
        self.p['fgw'] = fgw
        self.p['ggw'] = ggw
    
        V['vigw'] = vigw 
        V['vogw'] = vogw
        V['vhow'] = vhow
        V['vfgw'] = vfgw 
        V['vggw'] = vggw
    
        S['sigw'] = sigw 
        S['sogw'] = sogw
        S['show'] = show
        S['sfgw'] = sfgw 
        S['sggw'] = sggw
    
        return V,S
    
    def forget_gate_activations(self,data,fg_weights):
        fg_act = np.dot(data,fg_weights)
        return self.sigmoid(fg_act)
    
    def input_gate_activations(self,data,ig_weights):
        ig_act = np.dot(data,ig_weights)
        return self.sigmoid(ig_act)
    
    def output_gate_activations(self,data,og_weights):
        og_act = np.dot(data,og_weights)
        return self.sigmoid(og_act)
    
    def gate_activations(self,data,gg_weights):
        gg_act = np.dot(data,gg_weights)
        return self.tanh_activation(gg_act)
    
    def single_lstm_cell(self, batch_dataset, prev_activation_matrix, prev_cell_matrix):
        igw = self.p['igw']
        ogw = self.p['ogw']
        fgw = self.p['fgw']
        ggw = self.p['ggw']
    
        c_df = np.concatenate((batch_dataset,prev_activation_matrix),axis=1)
        
        fa = self.forget_gate_activations(c_df,fgw)
        ia = self.input_gate_activations(c_df,igw)
        oa = self.output_gate_activations(c_df,ogw)
        ga = self.gate_activations(c_df,ggw)
    
        cell_memory_matrix = np.multiply(fa,prev_cell_matrix) + np.multiply(ia,ga)
        activation_matrix = np.multiply(oa, self.tanh_activation(cell_memory_matrix))
    
        lstm_activations = dict()
        lstm_activations['fa'] = fa
        lstm_activations['ia'] = ia
        lstm_activations['oa'] = oa
        lstm_activations['ga'] = ga
    
        return lstm_activations,cell_memory_matrix,activation_matrix
    
    def single_output_cell(self,activation_matrix):
        how = self.p['how']
    
        output_matrix = np.dot(activation_matrix,how)
        output_matrix = self.softmax(output_matrix)
    
        return output_matrix
    
    def forward_propagation(self,bs,edg):
        b_len = bs[0].shape[0]
    
        lstm_c = dict()                 
        act_c = dict()           
        cell_c = dict()                 
        ot_c = dict()               
        edg_c = dict()            
    
        act0 = np.zeros([b_len,self.hidden_units],dtype=np.float32)
        cell0 = np.zeros([b_len,self.hidden_units],dtype=np.float32)
    
        act_c['a0'] = act0
        cell_c['c0'] = cell0
    
        for i in range(len(bs)-1):
            batch_dataset = bs[i]
        
            batch_dataset = self.get_edgs(batch_dataset,edg)
            edg_c['emb'+str(i)] = batch_dataset
        
            lstm_activations,ct,at = self.single_lstm_cell(batch_dataset,act0,cell0)
        
            ot = self.single_output_cell(at)
        
            lstm_c['lstm' + str(i+1)]  = lstm_activations
            act_c['a'+str(i+1)] = at
            cell_c['c' + str(i+1)] = ct
            ot_c['o'+str(i+1)] = ot
        
            act0 = at
            cell0 = ct
        
        return edg_c,lstm_c,act_c,cell_c,ot_c

    def backward_propagation(self,b_lbls,embedding_cache,lstm_cache,activation_cache,cell_cache,output_cache):
        output_error_cache,activation_error_cache = self.calculate_output_cell_error(b_lbls,output_cache)
    
        lstm_error_cache = dict()
        embedding_error_cache = dict()
        eat = np.zeros(activation_error_cache['ea1'].shape)
        ect = np.zeros(activation_error_cache['ea1'].shape)
        
        i = len(lstm_cache)
        while(i>0):
            pae,pce,ee,le = self.calculate_single_lstm_cell_error(activation_error_cache['ea'+str(i)],eat,ect,lstm_cache['lstm'+str(i)],cell_cache['c'+str(i)],cell_cache['c'+str(i-1)])
        
            lstm_error_cache['elstm'+str(i)] = le
            embedding_error_cache['eemb'+str(i-1)] = ee
            eat = pae
            ect = pce
            i = i - 1
    
    
        dts = dict()
        dts['dhow'] = self.calculate_output_cell_derivatives(output_error_cache,activation_cache)
    
        lstm_dts = dict()
        x = 0
        while(x < len(lstm_error_cache)):
            lstm_dts['dlstm'+str(x+1)] = self.calculate_single_lstm_cell_derivatives(lstm_error_cache['elstm'+str(x+1)],embedding_cache['emb'+str(x)],activation_cache['a'+str(x)])
            x = x + 1
    
        dts['dfgw'] = np.zeros(self.p['fgw'].shape)
        dts['digw'] = np.zeros(self.p['igw'].shape)
        dts['dogw'] = np.zeros(self.p['ogw'].shape)
        dts['dggw'] = np.zeros(self.p['ggw'].shape)
        
        y = 0
        while( y < len(lstm_error_cache)):
            dts['dfgw'] += lstm_dts['dlstm'+str(y+1)]['dfgw']
            dts['digw'] += lstm_dts['dlstm'+str(y+1)]['digw']
            dts['dogw'] += lstm_dts['dlstm'+str(y+1)]['dogw']
            dts['dggw'] += lstm_dts['dlstm'+str(y+1)]['dggw']
            y = y + 1  
    
        return dts,embedding_error_cache
    
    def calculate_loss_accuracy(self,b_lbls,ot_c):
        loss = 0  
        acc  = 0 
    
        b_len = b_lbls[0].shape[0]
        i = 0
        while( i < len(ot_c)):
            lbls = b_lbls[i+1]
            pred = ot_c['o'+str(i+1)]
            loss += -np.mean(np.log(pred)*lbls)
            acc  += np.array(np.argmax(lbls,1)==np.argmax(pred,1),dtype=np.float32).reshape(-1,1)
            i = i + 1
    
        acc  = np.sum(acc)/(b_len)
        acc = acc/len(ot_c)
    
        return loss,acc
    
    def calculate_output_cell_error(self,b_lbls,ot_c):
        ot_e_c = dict()
        act_e_c = dict()
        how = self.p['how']
        
        i = 0
        while( i < len(ot_c)):
            labels = b_lbls[i+1]
            pred = ot_c['o'+str(i+1)]
            error_output = pred - labels
            error_activation = np.dot(error_output,how.T)
            ot_e_c['eo'+str(i+1)] = error_output
            act_e_c['ea'+str(i+1)] = error_activation
            i = i + 1
        
        return ot_e_c,act_e_c
    
    def calculate_single_lstm_cell_error(self,activation_output_error,next_activation_error,next_cell_error,lstm_activation,cell_activation,prev_cell_activation):
        activation_error = activation_output_error + next_activation_error
    
        oa = lstm_activation['oa']
        eo = np.multiply(activation_error,self.tanh_activation(cell_activation))
        eo = np.multiply(np.multiply(eo,oa),1-oa)
    
        cell_error = np.multiply(activation_error,oa)
        cell_error = np.multiply(cell_error,self.tanh_derivative(self.tanh_activation(cell_activation)))
        cell_error += next_cell_error
    
        ia = lstm_activation['ia']
        ga = lstm_activation['ga']
        ei = np.multiply(cell_error,ga)
        ei = np.multiply(np.multiply(ei,ia),1-ia)
    
        eg = np.multiply(cell_error,ia)
        eg = np.multiply(eg,self.tanh_derivative(ga))
    
        fa = lstm_activation['fa']
        ef = np.multiply(cell_error,prev_cell_activation)
        ef = np.multiply(np.multiply(ef,fa),1-fa)
    
        prev_cell_error = np.multiply(cell_error,fa)
    
        fgw = self.p['fgw']
        igw = self.p['igw']
        ggw = self.p['ggw']
        ogw = self.p['ogw']
    
        embed_activation_error = np.dot(ef,fgw.T)
        embed_activation_error += np.dot(ei,igw.T)
        embed_activation_error += np.dot(eo,ogw.T)
        embed_activation_error += np.dot(eg,ggw.T)
    
        input_hidden_units = fgw.shape[0]
        self.hidden_units = fgw.shape[1]
        self.input_units = input_hidden_units - self.hidden_units
    
        prev_activation_error = embed_activation_error[:,self.input_units:]
    
        embed_error = embed_activation_error[:,:self.input_units]
    
        lt_e = dict()
        lt_e['ef'] = ef
        lt_e['ei'] = ei
        lt_e['eo'] = eo
        lt_e['eg'] = eg
    
        return prev_activation_error,prev_cell_error,embed_error,lt_e

    def calculate_output_cell_derivatives(self,output_error_cache,activation_cache):
        dhow = np.zeros(self.p['how'].shape)
    
        b_len = activation_cache['a1'].shape[0]
        
        i = 0
        while( i < len(output_error_cache)):
            output_error = output_error_cache['eo' + str(i+1)]
            activation = activation_cache['a'+str(i+1)]
            dhow += np.dot(activation.T,output_error)/b_len
            i = i + 1
        
        return dhow
    
    def calculate_single_lstm_cell_derivatives(self,lstm_error,edg_matrix,activation_matrix):
        ef = lstm_error['ef']
        ei = lstm_error['ei']
        eo = lstm_error['eo']
        eg = lstm_error['eg']
    
        concat_matrix = np.concatenate((edg_matrix,activation_matrix),axis=1)
    
        batch_size = edg_matrix.shape[0]
    
        dfgw = np.dot(concat_matrix.T,ef)/batch_size
        digw = np.dot(concat_matrix.T,ei)/batch_size
        dogw = np.dot(concat_matrix.T,eo)/batch_size
        dggw = np.dot(concat_matrix.T,eg)/batch_size
    
        derivatives = dict()
        derivatives['dfgw'] = dfgw
        derivatives['digw'] = digw
        derivatives['dogw'] = dogw
        derivatives['dggw'] = dggw
    
        return derivatives
    
    def get_edgs(self,b_df,edg):
        edg_df = np.dot(b_df,edg)
        return edg_df
    
    def update_edgs(self,edg,edg_e_c,b_lbls):
        edg_dts = np.zeros(edg.shape)
    
        b_len = b_lbls[0].shape[0]
    
        i = 0
        while( i < len(edg_e_c)):
            edg_dts += np.dot(b_lbls[i].T,edg_e_c['eemb'+str(i)])/b_len
            i = i + 1
    
        edg = edg - self.learning_rate*edg_dts
        
        return edg
    
    def train(self,batch_size=20):
        V = self.initialize_V()
        S = self.initialize_S()
    
        edg = np.random.normal(0,0.1,(self.vlength,self.input_units))
    
        for epoch in range(self.epochs+1):
            index = epoch%len(self.train_df)
            batches = self.train_df[index]
        
            edg_cache,lstm_cache,activation_cache,cell_cache,output_cache = self.forward_propagation(batches,edg)
        
            loss,acc = self.calculate_loss_accuracy(batches,output_cache)
        
            derivatives,edg_error_cache = self.backward_propagation(batches,edg_cache,lstm_cache,activation_cache,cell_cache,output_cache)
        
            V,S = self.update_parameters(derivatives,V,S,epoch)
        
            edg = self.update_edgs(edg,edg_error_cache,batches)
        
            if(epoch%100==0):
                print("Epoch       : {}".format(epoch))
                print("Loss       : {}".format(round(loss,3)))
                print("Accuracy   : {}".format(round(acc*100,3)))
                print("")
            
            if(epoch == self.epochs):
                print("Training Complete!")
                print("Loss at the end of training : {}".format(round(loss,3)))
                print("Accuracy at the end of training : {}".format(round(acc*100,3)))
        
        return edg
    
    def predict(self,edg,names_num=10):
        names = []
    
        for i in range(names_num):
            a0 = np.zeros([1,self.hidden_units],dtype=np.float32)
            c0 = np.zeros([1,self.hidden_units],dtype=np.float32)

            name = ''
            batch_dataset = np.zeros([1,self.vlength])       
            index = np.random.randint(0,27,1)[0]        
            batch_dataset[0,index] = 1.0        
            name += self.id_char[index]    
            char = self.id_char[index]
        
            while(char!=' '):
                batch_dataset = self.get_edgs(batch_dataset,edg)
                lstm_activations,ct,at = self.single_lstm_cell(batch_dataset,a0,c0)
                ot = self.single_output_cell(at)
                pred = np.argmax(ot)               
                name += self.id_char[pred]           
                char = self.id_char[pred]
            
                batch_dataset = np.zeros([1,self.vlength])
                batch_dataset[0,pred] = 1.0

                a0 = at
                c0 = ct
            
            names.append(name)
        
        return names
    
if __name__ == "__main__":
    print("Enter number of epochs to make: ")
    num = int(input())
    ml = Model('https://personal.utdallas.edu/~rxk190020/NationalNames.csv',num)
    embeddings = ml.train()
    print("Predictions: ")
    print("Enter number of names to predict: ")
    names_num = int(input())
    names = ml.predict(embeddings,names_num)
    for name in names:
        print(name)