<template>
  <main class="page-cascades">
    <div class="cont">
      <LoadModel v-model="dialogLoadModel" />
      <SaveModel v-model="dialogSaveModel" :image="imageModel" />
      <Toolbar @actions="actions" />
      <Blocks ref="container" />
      <Params />
      <CopyModal v-model="kerasModal" :title="'Код на keras'">{{ keras }}</CopyModal>
      <DatasetsModal v-model="dialogDatasets" />
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
import DatasetsModal from '../components/cascades/modals/DatasetsModal.vue'

export default {
  name: 'cascades',
  components: {
    Toolbar,
    Blocks,
    Params,
    LoadModel,
    SaveModel,
    CopyModal,
    DatasetsModal
  },
  data: () => ({
    dialogLoadModel: false,
    dialogSaveModel: false,
    dialogDatasets: false,
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
      this.dialogLoadModel = true// = await this.$store.dispatch('dialogs/trining', { ctx: this, page: 'модели' });
    },
    addBlock(type) {
      const position = this.$refs.container.getCenter();
      this.create = false;
      this.$store.dispatch('cascades/addBlock', { type, position });
    },
    async save() {
      this.imageModel = null;
      this.dialogSaveModel = true;
      let image = await this.$refs.container.getImages();
      const { data = null } = await this.$store.dispatch('cascades/getImageModel', image.slice(22));
      if (data) this.imageModel = data;
    },
    async start() {
      this.dialogDatasets = true
      await this.$store.dispatch('cascades/setDatasets')
      // await this.$store.dispatch('cascades/start', {});
    },
    async validation() {
      await this.$store.dispatch('cascades/validate', {});
    },
    async clearModel() {
      const action = await this.$store.dispatch('dialogs/confirm', { ctx: this, content: 'Очистить каскад?' });
      if (action == 'confirm') {
        await this.$store.dispatch('cascades/clearModel');
      }
    },
    actions(btn) {
      if (btn === 'load') {
        this.isTraining();
      }
      if (['InputData', 'Model', 'Function', 'Custom', 'Service', 'OutputData'].includes(btn)) {
        console.log(btn);
        this.addBlock(btn);
      }
      if (btn === 'start') {
        this.start();
      }
      if (btn === 'save') {
        this.save();
      }
      if (btn === 'validation') {
        this.validation();
      }
      if (btn === 'clear') {
        this.clearModel();
      }
      if (btn === 'keras') {
        this.kerasModal = true;
      }
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
