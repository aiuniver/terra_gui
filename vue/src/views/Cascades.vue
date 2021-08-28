<template>
  <main class="page-modeling">
    <div class="cont">
      <LoadModel v-model="dialogLoadModel" />
      <SaveModel v-model="dialogSaveModel" :image="imageModel"/>
      <Toolbar @actions="actions" />
      <Blocks ref="container" @blockSelect="selectBlock = $event" @blockDeselect="selectBlock = null" @save="saveLayers" />
      <Params ref="params" :selectBlock="selectBlock" />
    </div>
  </main>
</template>

<script>
import Toolbar from '@/components/cascades/Toolbar';
import Blocks from '@/components/cascades/block/Blocks';
import Params from '@/components/cascades/Params';
import LoadModel from '@/components/cascades/modals/LoadModel';
import SaveModel from '@/components/cascades/modals/SaveModel';

export default {
  name: 'Cascades',
  components: {
    Toolbar,
    Blocks,
    Params,
    LoadModel,
    SaveModel,
  },
  data: () => ({
    dialogLoadModel: false,
    dialogSaveModel: false,
    selectBlock: null,
    imageModel: null,
  }),
  methods: {
    addBlock(type) {
      console.log(type);
      this.create = false;
      this.selectBlockType = '';
      this.$refs.container.addNewBlock(type);
    },
    async saveModel() {
      this.imageModel = null
      this.dialogSaveModel = true;
      this.imageModel = await this.$refs.container.getImages();
    },
    async saveLayers() {
      await this.$store.dispatch("cascades/saveModel", {});
    },
    actions(btn) {
      if (btn === 'load') {
        this.dialogLoadModel = true;
      }
      if (btn === 'input' || btn === 'middle' || btn === 'output') {
        this.addBlock(btn);
      }
      if (btn === 'save') {
        this.saveModel()  
      }
      if (btn === 'validation') {
        // this.create = true
        console.log('hjkhjh')
        this.$refs.params.saveModel()
      }
      if (btn === 'clear') {
        this.$Modal.confirm({
          title: 'Внимание!',
          content: 'Очистить модель?',
          width: 300,
          callback: function (action) {
            // this.$Message(action)
            if(action == "confirm"){
              console.log("DELETE MODEL")
            }
          }
        })
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
