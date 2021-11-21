<template>
  <div class="card">
    <div class="card__content">
      <div v-if="type == 'ImageClassification'">
        <div class="card__original">
          <ImgCard :imgUrl="card.source" />
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '224px', height: '80px' }">{{ ClassificationResult }}</TextCard>
        </div>
      </div>
      <div v-if="type == 'TextClassification'">
        <div class="card__original">
          <TextCard :style="{ width: '600px', color: '#A7BED3', height: '324px' }">{{ card.source }}</TextCard>
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '600px', height: '80px' }">{{ ClassificationResult}}</TextCard>
        </div>
      </div>
      <div v-if="type == 'TextSegmentation'">
        <div class="card__original segmentation__original" :style="{ height: '324px' }">
          <scrollbar :ops="ops">
            <TableTextSegmented
              v-bind="{value: card.format, tags_color: {segmentationLayer}, layer: 'segmentationLayer', block_width: '598px'}"
              :key="RandId"
            />
          </scrollbar>
        </div>
        <div class="card__result">
          <SegmentationTags
            :style="{ width: '600px', height: '80px' }"
            :tags="segmentationLayer"
          />
        </div>
      </div>
      <div v-if="type == 'AudioClassification'">
        <div class="card__original">
          <AudioCard :value="card.source" :update="RandId" />
        </div>
        <div class="card__result">
          <TextCard :style="{ width: '600px', height: '80px' }">{{ ClassificationResult }}</TextCard>
        </div>
      </div>

      <div v-if="type == 'ImageSegmentation'">
        <div class="card__original">
          <ImgCard :imgUrl="card.source" />
        </div>
        <div class="card__result">
          <ImgCard :imgUrl="card.segment" />
        </div>
      </div>
      <div v-if="type == 'VideoObjectDetection'">
        <div class="card__original">
          <TableVideo :value="card.source" />
        </div>
        <div class="card__result">
          <TableVideo :value="card.predict" />
        </div>
      </div>
      <div v-if="type == 'YoloV3' || type == 'YoloV4'">
        <div class="card__original">
          <TableImage size="large" :value="card.source" />
        </div>
        <div class="card__result">
          <TableImage size="large" :value="card.predict" />
        </div>
      </div>
      <div class="card__graphic" v-if="type == 'Timeseries'">
        <GraphicCard v-bind="card" :key="'graphic_' + index"/>
      </div>
    </div>
    <div class="card__reload"><button class="btn-reload" @click="ReloadCard"><i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i></button></div>
  </div>
</template>

<script>
import ImgCard from './cards/ImgCard';
import TableVideo from './cards/TableVideo';
import TextCard from './cards/TextCard';
import AudioCard from './cards/AudioCard';
import TableTextSegmented from "../training/main/prediction/components/TableTextSegmented";
import SegmentationTags from "./cards/SegmentationTags";
import GraphicCard from "./cards/GraphicCard";
import { mapGetters } from 'vuex';
export default {
  name: 'IndexCard',
  components: {
    ImgCard,
    TableVideo,
    TextCard,
    GraphicCard,
    AudioCard,
    TableTextSegmented,
    SegmentationTags,
    TableImage: () => import('@/components/training/main/prediction/components/TableImage'),

  },
  data: () => ({
    ops: {
      scrollPanel: {
        scrollingX: false,
        scrollingY: true,
      },
    },
  }),
  props: {
    card: {
      type: Object,
      default: () => ({}),
    },
    index: [String, Number],
    color_map: {
      type: Array,
      default: () => ([]),
    }
  },

  methods: {
    ReloadCard() {
      this.$emit('reload', [this.index.toString()]);
    },
    GraphicData(){

    }
  },
  computed: {
    ...mapGetters({
      graphicData: 'deploy/getGraphicData',
      defaultLayout: 'deploy/getDefaultLayout',
      origTextStyle: 'deploy/getOrigTextStyle',
      type: 'deploy/getDeployType',
      RandId: 'deploy/getRandId',
    }),
    layout() {
      const layout = this.defaultLayout;
      if (this.char) {
        layout.title.text = this.char.title || '';
        layout.xaxis.title = this.char.xaxis.title || '';
        layout.yaxis.title = this.char.yaxis.title || '';
      }
      return layout;
    },
    segmentationLayer(){
      let layer = {}
      for(let i in this.color_map){
        if(this.color_map[i][0].includes("p")) continue;
        let tag = this.color_map[i][0].slice(1, this.color_map[i][0].length-1);
        layer[tag] = this.color_map[i][2];
      }
      // console.log(layer);
      return layer
    },
    ClassificationResult() {
      let text = this.card.data;
      let prepareText = '';
      text.sort((a, b) => (a[1] < b[1] ? 1 : -1));
      for (let i = 0; i < text.length; i++) {
        prepareText = prepareText + `${text[i][0]} - ${text[i][1]}% \n`;
      }
      return prepareText;
    },
  },
  mounted() {
    console.log(this.card)
  }
};
</script>

<style lang="scss" scoped>
.card__reload {
  padding-left: 5px;
}
.card {
  padding: 15px 15px 15px 0;
  display: flex;
}
.card__graphic {
  background: #242f3d;
  border: 1px solid #6c7883;
  box-sizing: border-box;
  border-radius: 4px;
}
.card__original {
  background: #242f3d;
}
.card__result {
  padding-top: 6px;
}
.btn-reload {
  width: 32px;
  height: 32px;
  i {
    position: absolute;
    margin-left: 7px;
    margin-top: -13px;
    width: 16px;
  }
}
.card__table {
  width: 100%;
}
.segmentation{
  &__original{
    border: 1px solid #6c7883;
    border-radius: 4px;
  }
}
</style>