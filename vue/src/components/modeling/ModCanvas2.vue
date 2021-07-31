<template>
  <div class="board">
    <div class="canvas" :style="height">
      <VueBlocksContainer
        ref="container"
        :scene.sync="scene"
        class="cont"
        @blockSelect="selected"
        @blockDeselect="selected(null)"
      />
    </div>
  </div>
</template>

<script>
import VueBlocksContainer from "@/components/modeling/block/VueBlocksContainer";
import { mapGetters } from 'vuex';
export default {
  name: "ModCanvas",
  components: {
    VueBlocksContainer,
  },
  data() {
    return {
      dialog: false,
      create: false,
      selectBlockType: '',
    };
  },
  computed: {
    ...mapGetters({
      height: "settings/autoHeight",
      toolbar: "modeling/getToolbarEvent",
      typeBlock: "modeling/getTypeBlock",
      blocks: "modeling/getBlocks",
    }),
    scene: {
      set(value) {
        this.$store.dispatch('modeling/setScene', value)
      },
      get() {
        return this.$store.getters["modeling/getScene"]
      }
    },
    select: {
      set(value) {
        this.$store.dispatch('modeling/setSelect', value)
      },
      get() {
        return this.$store.getters["modeling/getSelect"]
      }
    }
  },
  methods: {
    addBlock (type) {
      console.log(type)
      this.create = false
      this.selectBlockType = ''
      this.$refs.container.addNewBlock(type)
    },
    selected(block) {
      this.select = block?.id || null
    },
    doSomething(value) {
      console.log(value)
    }
  },
  watch: {
    toolbar: {
      handler({ event }) {
        if (event === 'middle') {
          this.addBlock(event)
        }
        if (event === 'validation') {
          this.$store.dispatch('test', 'sdsdsdsdsdsd')
          // this.create = true
        }
        console.log(event)
      }
    }
  }
};
</script>

<style lang="scss" scoped>
.board {
  flex-shrink: 1;
  position: relative;
  width: 100%;
}
.cont {
  height: 100%;
}
</style>