<template>
  <main class="page-modeling">
    <div class="cont">
      <LoadModel v-model="dialogLoadModel" />
      <SaveModel v-model="dialogSaveModel" :image="imageModel" />
      <Toolbar @actions="actions" />
      <Blocks ref="container" />
      <Params />
      <CopyModal v-model="kerasModal" :title="'Код на keras'">{{ keras }}</CopyModal>
    </div>
  </main>
</template>

<script>
import Toolbar from '@/components/modeling/Toolbar';
import Blocks from '@/components/modeling/block/Blocks';
import Params from '@/components/modeling/Params';
import LoadModel from '@/components/modeling/modals/LoadModel';
import SaveModel from '@/components/modeling/modals/SaveModel';
import CopyModal from '../components/global/modals/CopyModal';

export default {
  name: 'Modeling',
  components: {
    Toolbar,
    Blocks,
    Params,
    LoadModel,
    SaveModel,
    CopyModal,
  },
  data: () => ({
    dialogLoadModel: false,
    dialogSaveModel: false,
    imageModel: null,
    kerasModal: false,
  }),
  computed: {
    keras() {
      return this.$store.getters['modeling/getModel']?.keras || '';
    },
  },
  methods: {
    addBlock(type) {
      console.log(type);
      this.create = false;
      this.$store.dispatch('modeling/addBlock', type);
    },
    async saveModel() {
      this.imageModel = null;
      this.dialogSaveModel = true;
      this.imageModel = await this.$refs.container.getImages();
    },
    async validateModel() {
      const validate = await this.$store.dispatch('modeling/validateModel', {});
      if(Object.values(validate).some((el) => el == null)){
        this.$store.dispatch(
              'messages/setMessage',
              { message: `Модель прошла валидацию успешно!` },
              { root: true }
            );
      }
    },
    async clearModel() {
      try {
        const action = await this.$Modal.confirm({
          title: 'Внимание!',
          content: 'Очистить модель?',
          width: 300,
        });
        if (action == 'confirm') {
          console.log('DELETE MODEL');
          this.$store.dispatch('modeling/clearModel');
        }
      } catch (error) {
        console.log(error);
      }
    },
    actions(btn) {
      if (btn === 'load') {
        this.dialogLoadModel = true;
      }
      if (btn === 'input' || btn === 'middle' || btn === 'output') {
        this.addBlock(btn);
      }
      if (btn === 'save') {
        this.saveModel();
      }
      if (btn === 'validation') {
        this.validateModel();
      }
      if (btn === 'clear') {
        this.clearModel();
      }
      if (btn === 'keras') {
        this.kerasModal = true;
      }
      console.log(btn);
    },
  },
};
</script>

<style lang="scss" scoped>
.cont {
  background: #17212b;
  padding: 0;
  display: flex;
  height: 100%;
}
</style>
