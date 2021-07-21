<template>
  <div class="board">
    <div class="canvas" :style="height">
      <VueBlocksContainer
        ref="container"
        :blocksContent="blocks"
        :scene.sync="scene"
        class="cont"
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
      scene: {
        blocks: [
          {
            id: 1,
            position: [-900, 50],
            // x: -900,
            // y: 50,
            name: "input",
            title: "Input",
            parameters: {
              main: {
                x_cols: {
                  type: "string",
                  parse: "[main][x_cols]",
                  default: "asda",
                },
              },
              extra: {
                x_cols: {
                  type: "string",
                  parse: "[extra][x_cols]",
                  default: "a",
                },
              }
            }
          },
          {
            id: 2,
            position: [-900, 150],
            // x: -900,
            // y: 150,
            name: "sloy-one",
            title: "Sloy",
            parameters: {
              main: {
                x_cols: {
                  type: "string",
                  parse: "[main][x_cols]",
                  default: "ddd",
                },
              },
              extra: {
                x_cols: {
                  type: "string",
                  parse: "[extra][x_cols]",
                  default: "h",
                },
              }
            }
          },
          {
            id: 3,
            position: [-900, 250],
            // x: -900,
            // y: 250,
            name: "sloy-two",
            title: "Sloy",
            parameters: {
              main: {
                x_cols: {
                  type: "string",
                  parse: "[main][x_cols]",
                  default: "",
                },
              },
              extra: {
                x_cols: {
                  type: "string",
                  parse: "[extra][x_cols]",
                  default: "",
                },
              }
            }
          },
          {
            id: 4,
            position: [-900, 350],
            // x: -900,
            // y: 350,
            name: "sloy-three",
            title: "Sloy",
            parameters: {}
          },
          {
            id: 5,
            position: [-900, 450],
            // x: -900,
            // y: 450,
            name: "output",
            title: "Output",
            parameters: {
              main: {
                x_cols: {
                  type: "string",
                  parse: "[main][x_cols]",
                  default: "",
                },
              },
              extra: {
                x_cols: {
                  type: "string",
                  parse: "[extra][x_cols]",
                  default: "",
                },
              }
            }
          },
        ],
        links: [
          {
            id: 1,
            originID: 1,
            originSlot: 0,
            targetID: 2,
            targetSlot: 0,
          },
          {
            id: 2,
            originID: 2,
            originSlot: 1,
            targetID: 3,
            targetSlot: 0,
          },
          {
            id: 3,
            originID: 1,
            originSlot: 0,
            targetID: 3,
            targetSlot: 0,
          },
        ],
        container: {
          centerX: 1042,
          centerY: 140,
          scale: 1,
        },
      },
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
    })
  },
  methods: {
    addBlock (type) {
      console.log(type)
      this.create = false
      this.selectBlockType = ''
      this.$refs.container.addNewBlock(type)
    },
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
  position: relative;
}
.cont {
  height: 100%;
}
</style>