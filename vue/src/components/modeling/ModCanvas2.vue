<template>
  <div class="board">
    <div class="canvas" :style="height">
      <VueBlocksContainer
        ref="container"
        :blocksContent="blocks"
        :scene.sync="scene"
        class="cont"
        @blockSelect="selected"
        @blockDeselect="selected(null)"
      />
    </div>
    <at-modal v-model="create" width="150">
      <div slot="header">
        <span>Тип слоя</span>
      </div>
      <div>
        <at-dropdown @on-dropdown-command="addBlock">
          <at-button size="large" :style="{}" >Тип Слоя<i class="icon icon-chevron-down" /></at-button>
          <at-dropdown-menu slot="menu">
            <at-dropdown-item v-for="({ title, value }, i) of typeBlock" :value="value" :key="'menu' + i" :name="value">{{ title }}</at-dropdown-item>
          </at-dropdown-menu>
        </at-dropdown>
      </div>
      <div slot="footer" class="d-flex">
        <!-- <at-button @click="addBlock" type="primary">Создать</at-button> -->
        <!-- <at-button @click="create = false">Отменить</at-button> -->
      </div>
    </at-modal>
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
        if (event === 'sloy') {
          this.create = true
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