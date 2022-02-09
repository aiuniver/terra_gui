<template>
  <div class="page-modeling">
    <LoadModel v-model="dialogLoadModel" />
    <SaveModel v-model="dialogSaveModel" :image="imageModel" />
    <Toolbar @actions="actions" />
    <Blocks ref="container" />
    <Params />
    <CopyModal v-model="kerasModal" :title="'Код на keras'">{{ keras }}</CopyModal>
  </div>
</template>

<script>
import Toolbar from '@/components/modeling/Toolbar';
import Blocks from '@/components/modeling/block/Blocks';
import Params from '@/components/new/modeling/Params';
import LoadModel from '@/components/new/modeling/modals/LoadModel';
import SaveModel from '@/components/new/modeling/modals/SaveModel';
import CopyModal from '@/components/global/modals/CopyModal';

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
    async isTraining() {
      this.dialogLoadModel = await this.$store.dispatch('dialogs/trining', { ctx: this, page: 'модели' });
    },
    addBlock(type) {
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
      const action = await this.$store.dispatch('dialogs/confirm', { ctx: this, content: 'Очистить модель?' });
      if (action == 'confirm') {
        await this.$store.dispatch('modeling/clearModel');
      }
    },
    actions(btn) {
      if (btn === 'load') {
        this.isTraining();
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
.page-modeling {
  background: #17212b;
  padding: 0;
  display: flex;
  height: 100%;
}
</style>
