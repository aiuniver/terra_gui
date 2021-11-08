<template>
  <main class="page-cascades">
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
import Toolbar from '@/components/cascades/Toolbar';
import Blocks from '@/components/cascades/block/Blocks';
import Params from '@/components/cascades/Params';
import LoadModel from '@/components/cascades/modals/LoadModel';
import SaveModel from '@/components/cascades/modals/SaveModel';
import CopyModal from '../components/global/modals/CopyModal';

export default {
  name: 'cascades',
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
      return this.$store.getters['cascades/getModel']?.keras || '';
    },
  },
  methods: {
    async isTraining() {
      this.dialogLoadModel = await this.$store.dispatch('dialogs/trining', { ctx: this, page: 'модели' });
    },
    addBlock(type) {
      const position = this.$refs.container.getCenter();
      this.create = false;
      this.$store.dispatch('cascades/addBlock', { type, position });
    },
    async saveModel() {
      this.imageModel = null;
      this.dialogSaveModel = true;
      let image = await this.$refs.container.getImages();
      const { data = null } = await this.$store.dispatch('cascades/getImageModel', image.slice(22));
      if (data) this.imageModel = data;
    },
    async validateModel() {
      await this.$store.dispatch('cascades/validateModel', {});
    },
    async clearModel() {
      const action = await this.$store.dispatch('dialogs/confirm', { ctx: this, content: 'Очистить модель?' });
      if (action == 'confirm') {
        await this.$store.dispatch('cascades/clearModel');
      }
    },
    actions(btn) {
      if (btn === 'load') {
        this.isTraining();
      }
      if (['InputData', 'model', 'function', 'custom', 'OutputData'].includes(btn)) {
        this.addBlock(btn);
      }
      if (btn === 'save') {
        this.saveModel();
        this.$store.dispatch('cascades/selectBlock', {});
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
