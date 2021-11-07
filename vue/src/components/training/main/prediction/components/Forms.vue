<template>
  <div>
    <TableImage v-if="type === 'image'" v-bind="obj" :key="update" />
    <TableStr v-if="type === 'str' || type === 'number' " v-bind="obj" :key="update" />
    <TableStatisticTable v-if="type === 'table'" v-bind="obj" :key="update" />
    <TableText v-if="type === 'text'" v-bind="obj" :key="update" />
    <TableTextSegmented v-if="type === 'segmented_text'" v-bind="obj" :key="update" />
    <TableVideo v-if="type === 'video'" v-bind="obj" :key="update" />
    <TableAudio v-if="type === 'audio'" v-bind="obj" :key="update" />
    <Graphics v-if="type === 'graphic'" v-bind="obj" :key="update" />
  </div>
</template>

<script>
export default {
  name: 't-forms-prediction',
  components: {
    TableImage: () => import('./TableImage'),
    TableAudio: () => import('./TableAudio'),
    TableText: () => import('./TableText'),
    TableStr: () => import('./TableStr'),
    TableTextSegmented: () => import('./TableTextSegmented'),
    TableStatisticTable: () => import('./TableStatisticTable'),
    TableVideo: () => import('./TableVideo.vue'),
    Graphics: () => import('./TableGraphics'),
  },
  props: {
    type: {
      type: String,
      default: '',
    },
    data: {
      type: Object,
      default: () => {},
    },
    tags_color: {
      type: Object,
      default: () => {},
    },
    layer: {
      type: String,
      default: '',
    },
    update: {
      type: String,
      default: '',
    },
  },
  computed: {
    obj() {
      return { ...this.data, update: this.update, tags_color: this.tags_color, layer: this.layer };
    },
  },
};
</script>

<style lang="scss" scoped>
</style>