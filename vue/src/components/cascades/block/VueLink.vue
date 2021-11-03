<template>
  <svg width="100%" height="100%">
    <g v-for="(p,i) in renderedPathes" :key="'k' + i">
      <path v-if="outline" :d="p.data" :style="p.outlineStyle"></path>
      <circle v-if="!outline" :cx="p.circle[0]" :cy="p.circle[1]" :r="3 * p.scale" :fill="p.style.stroke"/>
      <circle v-if="!outline" :cx="p.circle[2]" :cy="p.circle[3]" :r="3 * p.scale" :fill="p.style.stroke"/>
      <path :d="p.data" :style="p.style"></path>
    </g>
    <g>
      <path v-for="(a, i) in renderedArrows" :key="'a' + i"
        d="M -1 -1 L 0 1 L 1 -1 z"
        :style="a.style"
        :transform="a.transform"></path>
    </g>
  </svg>
</template>

<script>
  export default {
    props: {
      lines: {
        type: Array,
        default () {
          return []
        }
      },
      outline: {
        type: Boolean,
        default: false
      }
    },
    methods: {
      distance (x1, y1, x2, y2) {
        return Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
      },
      computeConnectionPoint (x1, y1, x2, y2, t) {
        let dist = this.distance(x1, y1, x2, y2)
        let p0 = {x: x1, y: y1}
        let p1 = {x: x1 + dist * 0.25, y: y1}
        let p2 = {x: x2 - dist * 0.25, y: y2}
        let p3 = {x: x2, y: y2}

        let c1 = (1 - t) * (1 - t) * (1 - t)
        let c2 = 3 * ((1 - t) * (1 - t)) * t
        let c3 = 3 * (1 - t) * (t * t)
        let c4 = t * t * t

        let x = c1 * p0.x + c2 * p1.x + c3 * p2.x + c4 * p3.x
        let y = c1 * p0.y + c2 * p1.y + c3 * p2.y + c4 * p3.y
        return {x: x, y: y}
      }
    },
    computed: {
      renderedPathes () {
        if (!this.lines) {
          return []
        }

        let pathes = []
        this.lines.forEach(l => {
          // console.log(l)
          let dist = this.distance(l.x1, l.y1, l.x2, l.y2) * 0.2
          const circle = [l.x1, l.y1, l.x2, l.y2]
          const data  = [
            `M ${l.x1}, ${l.y1} C ${(l.x1)}, ${l.y1 + dist }, ${(l.x2)}, ${l.y2 - dist}, ${l.x2}, ${l.y2}`,
            `M ${l.x1}, ${l.y1} C ${(l.x1 + dist)}, ${l.y1}, ${(l.x2 - dist)}, ${l.y2}, ${l.x2}, ${l.y2}`,
            `M ${l.x1}, ${l.y1} C ${(l.x1 - dist)}, ${l.y1}, ${(l.x2 + dist)}, ${l.y2}, ${l.x2}, ${l.y2}`
          ]

          pathes.push({
            data: data[l.slot] || data[0],
            circle,
            scale: l.scale || 1,
            style: l.style,
            outlineStyle: l.outlineStyle
          })
        })

        return pathes
      },
      renderedArrows () {
        if (!this.lines) {
          return []
        }

        let arrows = []
        this.lines.forEach(l => {
          // l.x1, l.y1, l.x2, l.y2
          let pos = this.computeConnectionPoint(l.x1, l.y1, l.x2, l.y2, 0.5)
          let pos2 = this.computeConnectionPoint(l.x1, l.y1, l.x2, l.y2, 0.501)

          let angle = -Math.atan2(pos2.x - pos.x, pos2.y - pos.y)
          let degrees = (angle >= 0 ? angle : (2 * Math.PI + angle)) * 180 / Math.PI

          // console.log(l)

          arrows.push({
            transform: `translate(${pos.x}, ${pos.y}) rotate(${degrees - 15})`,
            style: {
              stroke: "rgb(80, 125, 150)",
              strokeWidth: l.style.strokeWidth * 2,
              fill: l.style.stroke
            }
          })
        })

        return arrows
      }
    }
  }
</script>

<style scoped>

</style>
