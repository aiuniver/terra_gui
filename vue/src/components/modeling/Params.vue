<template>
  <div class="params">
    <Navbar />
    <scrollbar>
      <div class="params__items">
        <div class="params__items--item">
          <t-input
            v-model="block.name"
            :label="'Название слоя'"
            :type="'text'"
            :parse="'name'"
            :name="'name'"
            :disabled="isBlock"
            @change="saveModel"
          />
          <Autocomplete2
            :value="block.typeLabel"
            :list="listWithoutOutputInput"
            label="Тип слоя"
            name="type"
            :disabled="isBlock || isInputOutput"
            @change="changeType"
          />
        </div>
        <at-collapse :value="collapse">
          <at-collapse-item v-show="main.items.length" class="mb-3" title="Параметры слоя">
            <Forms :data="main" @change="change" />
          </at-collapse-item>
          <at-collapse-item v-show="extra.items.length" class="mb-3" title="Дополнительные параметры">
            <Forms :data="extra" @change="change" />
          </at-collapse-item>
          <at-collapse-item v-show="!isBlock" class="mb-3" title="Размерность слоя" notChange>
            <Shape
              v-if="block.shape && block.shape.input"
              v-model="block.shape.input"
              :label="'Размерность входных данных'"
              :name="'shape_input'"
              :disabled="block.type !== 'Input' || !!project.dataset"
              @change="saveModel"
            />
            <Shape
              v-if="block.shape && block.shape.output"
              :value="block.shape.output"
              :label="'Размерность выходных данных'"
              :name="'shape_output'"
              :disabled="true"
            />
          </at-collapse-item>
        </at-collapse>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import Navbar from '@/components/modeling/comp/Navbar.vue';
import Shape from '@/components/forms/Shape.vue';
import Autocomplete2 from '@/components/forms/Autocomplete2.vue';
import Forms from '@/components/modeling/comp/Forms.vue';
import { mapGetters } from 'vuex';
// import serialize from "@/assets/js/serialize";

// import Select from "@/components/forms/Select.vue";
export default {
  name: 'Params',
  components: {
    Shape,
    Autocomplete2,
    Forms,
    Navbar,
    // Select
  },
  data: () => ({
    collapse: ['0', '2'],
    oldBlock: null,
  }),
  computed: {
    ...mapGetters({
      list: 'modeling/getList',
      layers: 'modeling/getLayersType',
      buttons: 'modeling/getButtons',
      block: 'modeling/getBlock',
      project: 'projects/getProject',
    }),
    isBlock() {
      return !this.block.id;
    },
    isInputOutput() {
      return this.block.group === 'input' || this.block.group === 'output';
    },
    listWithoutOutputInput() {
      return this.list.filter(item => !(item.value.toLowerCase() === 'input' || item.value.toLowerCase() === 'dense'));
    },

    buttonSave() {
      return this.buttons?.save || false;
    },
    main() {
      const blockType = this.block?.type;
      if (Object.keys(this.layers).length && blockType) {
        const items = this.layers[blockType]?.main || [];
        const value = this.block?.parameters?.main || {};
        return { type: 'main', items, value, blockType };
      } else {
        return { type: 'main', items: [], value: {} };
      }
    },
    extra() {
      const blockType = this.block?.type;
      if (Object.keys(this.layers).length && blockType) {
        const items = this.layers[blockType]?.extra || [];
        const value = this.block?.parameters?.extra || {};
        return { type: 'extra', items, value, blockType };
      } else {
        return { type: 'extra', items: [], value: {} };
      }
    },
  },
  methods: {
    async saveModel() {
      await this.$store.dispatch('modeling/updateModel', {});
    },
    async changeType({ value }) {
      await this.$store.dispatch('modeling/typeBlock', { type: value, block: this.block });
    },
    async change({ type, name, value }) {
      console.group();
      console.log({ type, name, value });
      console.log(this.collapse);
      console.groupEnd();
      if (this.block.parameters) {
        this.block.parameters[type][name] = value;
      } else {
        this.oldBlock.parameters[type][name] = value;
      }
      this.$emit('change');
      this.saveModel();
    },
  },
  watch: {
    block: {
      handler(newBlock, oldBlock) {
        this.oldBlock = oldBlock;
        // this.$store.dispatch('modeling/setSelect', newBlock?.id);
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
