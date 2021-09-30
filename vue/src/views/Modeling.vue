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
    isNoTrain() {
      return this.$store.getters['trainings/getStatus'] === 'no_train';
    },
  },
  methods: {
    async message() {
      await this.$store.dispatch('messages/setModel', {
        context: this,
        content: 'Для загрузки другой модели остановите обучение',
      });
    },
    addBlock(type) {
      // console.log(type);
      // console.log(this.$refs.container.centerX);
      const position = this.$refs.container.getCenter();
      this.create = false;
      this.$store.dispatch('modeling/addBlock', { type, position });
    },
    async saveModel() {
      this.imageModel = null;
      this.dialogSaveModel = true;
      let image = await this.$refs.container.getImages();
      const { data = null } = await this.$store.dispatch('modeling/getImageModel', image.slice(22));
      if (data) this.imageModel = data;
    },
    async validateModel() {
      await this.$store.dispatch('modeling/validateModel', {});
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
        if (this.isNoTrain) {
          this.dialogLoadModel = true;
        } else {
          this.message()
        }
      }
      if (btn === 'input' || btn === 'middle' || btn === 'output') {
        this.addBlock(btn);
      }
      if (btn === 'save') {
        this.saveModel();
        this.$store.dispatch('modeling/selectBlock', {});
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
