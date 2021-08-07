<template>
  <main class="page-modeling">
    <div class="cont">
      <ModalLoadModel />
      <Toolbar @actions="actions" />
      <Blocks
        ref="container"
        @blockSelect="selectBlock = $event"
        @blockDeselect="selectBlock = null"
      />
      <Params :selectBlock="selectBlock" @change="saveBlock"/>
    </div>
  </main>
</template>

<script>
import Toolbar from "@/components/modeling/Toolbar";
import Blocks from "@/components/modeling/block/Blocks";
import Params from "@/components/modeling/Params";
import ModalLoadModel from "@/components/modeling/ModalLoadModel";

export default {
  name: "Modeling",
  components: {
    Toolbar,
    Blocks,
    Params,
    ModalLoadModel,
  },
  data: () => ({
    selectBlock: null
  }),
  methods: {
    addBlock(type) {
      console.log(type);
      this.create = false;
      this.selectBlockType = "";
      this.$refs.container.addNewBlock(type);
    },
    async saveBlock() {
      console.log('saveBlock')
      await this.$store.dispatch("modeling/saveModel", {});
    },
     actions(btn) {
      if (btn === "middle") {
        this.addBlock(btn);
      }
      // if (btn === "save") {
         
      // }
      if (btn === "validation") {
        console.log(this.$refs.container.getImages());

        // this.create = true
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
