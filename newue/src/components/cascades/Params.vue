<template>
  <div class="params">
    <scrollbar>
      <div class="params__items">
        <div class="params__items--item">
          <t-field :label="'Название блока'">
            <DInputText
              v-model="block.name"
              :type="'text'"
              :parse="'name'"
              :name="'name'"
              :disabled="isBlock"
              @change="saveModel"
            />
          </t-field>
        </div>
        <at-collapse :value="collapse">
          <at-collapse-item v-show="main.length" class="mb-3" title="Параметры блока">
            <!-- <Forms :data="main" :id="block.id" @change="change" /> -->
            <template v-for="(data, i) of main">
              <t-auto-field-cascade
                v-bind="data"
                :key="data.name + i"
                :id="data.name"
                :parameters="parameters"
                :inline="false"
                @change="change"
              />
            </template>
          </at-collapse-item>
          <at-collapse-item v-show="!!info" class="mb-3" title="Информация о блоке">
            <BlockInfo :info="info" />
          </at-collapse-item>
          <!-- <at-collapse-item v-show="extra.items.length" class="mb-3" title="Дополнительные параметры">
            <Forms :data="extra" :id="block.id" @change="change" />
          </at-collapse-item> -->
        </at-collapse>
      </div>
    </scrollbar>
  </div>
</template>

<script>
// import Input from '@/components/forms/Input.vue';
import { mapGetters } from 'vuex';
import BlockInfo from '@/components/cascades/comp/Info.vue';
import { debounce } from '@/utils/core/utils';
export default {
  name: 'Params',
  components: {
    // Input,
    BlockInfo,
  },
  data: () => ({
    collapse: ['0', '1', '2'],
    oldBlock: null,
    debounce: null,
    example_count: null,
  }),
  computed: {
    ...mapGetters({
      list: 'cascades/getList',
      layers: 'cascades/getLayersType',
      layersForm: 'cascades/getLayersForm',
      // buttons: 'cascades/getButtons',
      block: 'cascades/getBlock',
      project: 'projects/getProject',
      manual: 'cascades/getManual',
    }),
    datatypes() {
      return this.layersForm.filter(({ name }) => name === `datatype_${this.block.group}`);
    },
    isBlock() {
      return !this.block.id;
    },
    isInput() {
      return this.block.group === 'input';
    },
    listWithoutOutputInput() {
      if (!this.list) return [];
      return this.list.filter(item => !(item.value.toLowerCase() === 'input'));
    },
    buttonSave() {
      return this.buttons?.save || false;
    },
    parameters() {
      return this.block?.parameters?.main || {};
    },
    main() {
      const blockType = this.block?.group;
      if (Object.keys(this.layers).length && blockType) {
        const items = this.layers[blockType]?.main || [];

        return items;
        // const value = this.block?.parameters?.main || {};
        //   return { type: 'main', items, value, blockType };
      } else {
        return [];
      }
    },
    info() {
      if (this.manual[this.block.group]) return this.manual[this.block.group][this.parameters.type];
      return '';
    },
    // extra() {
    //   const blockType = this.block?.group;
    //   if (Object.keys(this.layers).length && blockType) {
    //     const items = this.layers[blockType]?.extra || [];
    //     const value = this.block?.parameters?.extra || {};
    //     return { type: 'extra', items, value, blockType };
    //   } else {
    //     return { type: 'extra', items: [], value: {} };
    //   }
    // },
  },
  methods: {
    async changeId(value) {
      await this.$store.dispatch('cascades/changeId', value);
    },
    async saveModel() {
      await this.$store.dispatch('cascades/updateModel', this.block);
    },
    async changeType({ value }) {
      await this.$store.dispatch('cascades/typeBlock', { type: value, block: this.block });
    },
    async change({ value, name, mounted, parse }) {
      console.log('params change', value, name);
      const type = parse.split('][')[1].split(']')[0];
      if (this.block.parameters) {
        this.block.parameters['main'][type] = value;
      } else {
        this.oldBlock.parameters['main'][type] = value;
      }
      this.debounce(mounted);
    },
  },
  created() {
    this.debounce = debounce(() => {
      // console.log(status)
      this.saveModel();
    }, 200);
  },
  beforeDestroy() {
    this.debounce(false);
  },
  watch: {
    block: {
      handler(newBlock, oldBlock) {
        this.oldBlock = oldBlock;
        // this.$store.dispatch('cascades/setSelect', newBlock?.id);
        console.log(newBlock, oldBlock);
      },
    },
  },
};
</script>

<style lang="scss" scoped>
.params {
  width: 400px;
  flex-shrink: 0;
  border-left: #0e1621 solid 1px;
  overflow: hidden;
  height: 100%;
  // border-left: #0e1621  1px solid;
  &__items {
    height: 100%;
    padding-bottom: 20px;
    &--item {
      padding: 20px;
    }
  }
}
.params-actions {
  padding: 20px 10px;
}
</style>
